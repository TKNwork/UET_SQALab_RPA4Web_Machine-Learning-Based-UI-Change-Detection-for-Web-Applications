#!/usr/bin/env python3
"""Content correspondence -> change evidence -> frozen prediction coverage.

Only inventory/evaluate/report read GT. infer() accepts detections and images only.
No content-specific changes to the geometry classifier or its defaults.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from shared import matching as shared
from shared.regions import area, intersect, union_area, enclosure
from shared.paths import resource_path, groundtruth_root

ROOT = Path(__file__).resolve().parent.parent
DEV_IDS = ['187', '418', '437', '451', '489']
CONFIG = {
    'version': 'content-v1',
    'max_distance_diagonal': .60, 'max_size_ratio': 12.,
    'position_weight': 1.2, 'size_weight': .8, 'iou_weight': .5,
    'position_floor_width': 40., 'position_floor_height': 64., 'texture_weight': .6,
    'text_weight': .35, 'appearance_weight': .5, 'level_penalty': .4,
    'container_bonus': .4, 'exact_text_bonus': .8,
    'unmatched_cost': 3., 'max_match_cost': 3., 'stability_ban_cost_gap': .12,
    'pixel_difference': 30, 'visual_fraction': .12, 'visual_mae': 10.,
    'ocr_noise_similarity': .90, 'ocr_min_visual_fraction': .025,
    'max_prediction_image_fraction': .15,
    'group_gap_px': 32., 'group_motion_px': 24.,
    'coverage_gt_fraction': .5, 'coverage_support_fraction': .25,
}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def inventory():
    """GT selects the evaluation cohort, never correspondence/predictions."""
    if not any(groundtruth_root().glob('*.json')):
        raise FileNotFoundError('Content requires original flat bboxes GT files; configure RPA_GROUNDTRUTH. '
                                'Nested before_bboxes/after_bboxes archives use a different protocol.')
    labels = Counter(); samples = []; inputs = []
    for gt in sorted(groundtruth_root().glob('*.json')):
        data = read(gt)
        if 'bboxes' not in data:
            raise ValueError(f'Content requires bboxes GT schema, not paired/native GT: {gt.name}')
        for g in data['bboxes']:
            labels['+'.join(sorted(g['type']))] += 1
        selected = [i for i,g in enumerate(data['bboxes']) if 'content' in g['type']]
        inputs.append({'path':str(gt), 'sha256':sha(gt)})
        if not selected:
            continue
        pid = gt.stem
        b = sorted(resource_path('baseline', ROOT/'baseline/json').glob(pid+'_*'))
        a = sorted(resource_path('inject', ROOT/'inject/json').glob(pid+'_*'))
        images = sorted(resource_path('dataset1', ROOT.parent/'dataset1').glob(pid+'_*'))
        if len(b)!=1 or len(a)!=1 or len(images)!=1:
            raise ValueError(f'Ambiguous or missing source mapping: {pid}')
        if b[0].name != a[0].name or b[0].stem.removesuffix('_full_hierarchy') != images[0].name:
            raise ValueError(f'Source stem mismatch: {pid}')
        source = {'before':str(b[0]), 'after':str(a[0]), 'before_image':str(images[0]/'baseline.png'),
                  'after_image':str(images[0]/'inject.png'), 'groundtruth':str(gt)}
        sizes = []
        for side in ['before_image','after_image']:
            with Image.open(source[side]) as image:sizes.append(list(image.size))
        if sizes[0]!=sizes[1]:raise ValueError(f'Viewport mismatch {pid}')
        samples.append({'pair_id':pid, 'split':'dev' if pid in DEV_IDS else 'remaining',
                        'content_gt_indices':selected, 'content_gt_count':len(selected),
                        'multi_label_count':sum(len(data['bboxes'][i]['type'])>1 for i in selected),
                        'fallback_count':sum(data['bboxes'][i].get('fallback',False) for i in selected),
                        'image_size':sizes[0], 'sources':source})
        inputs += [{'path':p,'sha256':sha(p)} for p in source.values() if p!=str(gt)]
    assert [s['pair_id'] for s in samples[:5]]==DEV_IDS, 'Content cohort changed; choose and record new dev IDs'
    return {'multi_label_content_gt':sum(s['multi_label_count'] for s in samples),
            'multi_label_content_samples':sum(s['multi_label_count']>0 for s in samples),
            'fallback_content_gt':sum(s['fallback_count'] for s in samples),
            'selection':'all GT files; content in type; fixed dev = first five numeric IDs',
            'development_ids':DEV_IDS, 'source_gt_files':len(list(groundtruth_root().glob('*.json'))),
            'label_combinations':dict(labels), 'samples':samples,
            'content_samples':len(samples), 'content_gt':sum(s['content_gt_count'] for s in samples),
            'inputs':inputs}

def content_cost(b,a,diagonal,anchors,cfg,bh=None,ah=None,return_detail=False):
    """Soft OCR evidence, including OCR/non-OCR transitions. Never gates on text."""
    dist=shared.center_dist(b,a)
    if dist/diagonal>cfg['max_distance_diagonal'] or shared.size_ratio(b,a)>cfg['max_size_ratio']:
        return None
    nearby=sorted(anchors.get('local_samples',[]),key=lambda s:math.hypot(s['x']-b['cx'],s['y']-b['cy']))[:5]
    nearby=[s for s in nearby if math.hypot(s['x']-b['cx'],s['y']-b['cy'])<diagonal*.2]
    dx=float(np.median([s['dx'] for s in nearby])) if nearby else 0.
    dy=float(np.median([s['dy'] for s in nearby])) if nearby else 0.
    # Anchor-consistent displacement in units of the element's own dimensions.
    pos=math.hypot((a['cx']-b['cx']-dx)/max(b['width'],a['width'],cfg['position_floor_width']),
                   (a['cy']-b['cy']-dy)/max(b['height'],a['height'],cfg['position_floor_height']))
    text_sim=SequenceMatcher(None,b['ocr_text'],a['ocr_text']).ratio() if b['has_ocr'] and a['has_ocr'] else None
    costs={'position':cfg['position_weight']*min(pos,4),
           'size':cfg['size_weight']*shared.log_size_diff(b,a),
           'iou':cfg['iou_weight']*(1-shared.bbox_iou(b['bbox'],a['bbox'])),
           'level_transition':cfg['level_penalty']*int(b['has_ocr']!=a['has_ocr'])}
    if text_sim is not None:
        costs['text']=cfg['text_weight']*(1-text_sim)
        if text_sim==1:costs['exact_text']=-cfg['exact_text_bonus']
    crop_sim=shared.crop_histogram_sim_precomputed((bh or {}).get(b['id']),(ah or {}).get(a['id']))
    if crop_sim is not None:costs['appearance']=cfg['appearance_weight']*(1-crop_sim)
    textures=anchors.get('textures',{})
    bt=textures.get('before',{}).get(b['id']);at=textures.get('after',{}).get(a['id'])
    if bt is not None and at is not None:
        costs['texture']=cfg['texture_weight']*min(float(np.abs(np.asarray(bt)-np.asarray(at)).mean())/80,1)
    parent=next((p for p in anchors.get('container_candidates',[]) if p['before_id']==b['id'] and p['after_id']==a['id']),None)
    if parent:costs['container']=-cfg['container_bonus']*parent['evidence']['anchor_score']
    detail={'cost':sum(costs.values()),'components':costs,'text_similarity':text_sim,
            'local_anchor_count':len(nearby),'expected_shift':[dx,dy],
            'container_evidence':parent['evidence'] if parent else None,
            'confidence_before':b['confidence'],'confidence_after':a['confidence']}
    return detail if return_detail else detail['cost']

def crop_evidence(b,a,bi,ai,cfg):
    """Aligned grayscale texture difference; brightness/color alone is insufficient.

    Sample both element crops at native scale and search +-3 px translation.
    This is appearance evidence, not semantic recognition or a style classifier.
    """
    boxes=[intersect(e['bbox'],[0,0,im.width,im.height]) for e,im in [(b,bi),(a,ai)]]
    if any(area(bb)<8 for bb in boxes):
        return {'available':False,'reason':'outside_viewport_or_tiny'}
    # Use the SAME physical sampling scale for both endpoints. Independently
    # stretching a 12px and a 14px detector crop produces artificial text changes.
    width=min(768, max(round(bb[2]-bb[0]) for bb in boxes)+8)
    height=min(192, max(round(bb[3]-bb[1]) for bb in boxes)+8)
    arrays=[]
    for bb,im in zip(boxes,[bi,ai]):
        cx=round((bb[0]+bb[2])/2);cy=round((bb[1]+bb[3])/2)
        x0=cx-width//2;y0=cy-height//2
        crop=im.crop((x0,y0,x0+width,y0+height)).convert('L').filter(ImageFilter.GaussianBlur(.5))
        ar=np.asarray(crop,dtype=float);arrays.append(ar-np.median(ar))
    b_ar,a_ar=arrays;best=None
    for dy in range(-3,4):
        for dx in range(-3,4):
            x=b_ar[3:height-3,3:width-3];y=a_ar[3+dy:height-3+dy,3+dx:width-3+dx]
            diff=np.abs(x-y);value=(float(diff.mean()),float((diff>cfg['pixel_difference']).mean()),dx,dy)
            if best is None or value[0]<best[0]:best=value
    return {'available':True,'mae':best[0],'changed_fraction':best[1],
            'alignment_offset':[best[2],best[3]],'method':'native-scale common canvas median-centered blur0.5 translation3',
            'canvas_size':[width,height]}

def canonical(text):
    return ''.join(c for c in text.casefold() if c.isalnum())

def contained_text(e,raw):
    es=[r for r in raw if r['has_ocr'] and shared.containment(r['bbox'],e['bbox'])>=.8]
    return ' '.join(r['ocr_text'] for r in sorted(es,key=lambda r:(round(r['cy']/12),r['cx'])))

def classify(b,a,bi,ai,braw,araw,cfg):
    visual=crop_evidence(b,a,bi,ai,cfg)
    bt=b['ocr_text'] or contained_text(b,braw);at=a['ocr_text'] or contained_text(a,araw)
    ratio=SequenceMatcher(None,canonical(bt),canonical(at)).ratio() if bt and at else None
    flags=[]
    if shared.size_ratio(b,a)>2:flags.append('resize_or_segmentation_risk')
    if b['has_ocr']!=a['has_ocr']:flags.append('ocr_availability_transition')
    if any(e['confidence'] is not None and e['confidence']<.5 for e in [b,a]):flags.append('low_detector_confidence')
    state='unchanged';kind='none';reason='no_sufficient_content_evidence'
    if bt and at:
        if canonical(bt)==canonical(at):reason='same_text_including_split_merge_normalization'
        elif ratio>=cfg['ocr_noise_similarity']:
            state='uncertain';reason='small_text_difference_possible_ocr_noise'
        elif visual.get('changed_fraction',0)>=cfg['ocr_min_visual_fraction']:
            state='changed';kind='ocr_and_visual';reason='different_text_supported_by_changed_pixels'
        else:state='uncertain';reason='ocr_difference_without_visual_confirmation'
    elif visual.get('changed_fraction',0)>=cfg['visual_fraction'] and visual.get('mae',0)>=cfg['visual_mae']:
        state='changed';kind='visual_content_candidate';reason='spatial_texture_change_without_two_sided_ocr';flags.append('semantic_content_vs_style_requires_review')
    if max(b['area'],a['area'])/(bi.width*bi.height)>cfg['max_prediction_image_fraction'] and state=='changed':
        state='uncertain';reason='large_container_mixes_events'
    return {'state':state,'kind':kind,'reason':reason,'before_text':bt,'after_text':at,
            'text_similarity':ratio,'visual':visual,'quality_flags':flags,'human_correct':None}

def groups(predictions,cfg):
    """Complete-link local clusters of changed pairs; exact endpoint union, no gap filling."""
    result=[];seen=set()
    for seed in range(len(predictions)):
        ids=[seed]
        for i,p in enumerate(predictions):
            if i in ids:continue
            def compatible(j):
                q=predictions[j]
                if math.hypot(p['delta']['dx']-q['delta']['dx'],p['delta']['dy']-q['delta']['dy'])>cfg['group_motion_px']:return False
                for side in ['before_bbox','after_bbox']:
                    a,b=p[side],q[side]
                    gap=math.hypot(max(0,a[0]-b[2],b[0]-a[2]),max(0,a[1]-b[3],b[1]-a[3]))
                    if gap>cfg['group_gap_px']:return False
                return True
            if all(compatible(j) for j in ids):ids.append(i)
        key=tuple(sorted(ids))
        if len(ids)>1 and key not in seen:
            seen.add(key);result.append({'group_id':f'local_{len(result)}','member_prediction_indices':list(key),
                                        'support_boxes':[p[s] for i in key for p in [predictions[i]] for s in ['before_bbox','after_bbox']],
                                        'human_correct':None})
    return result

def infer(before_path,after_path,before_image,after_image,cfg=None):
    """GT-free API, with no pair ID, GT labels or selectors in its arguments."""
    cfg=cfg or CONFIG;bcfg=dict(shared.NORMALIZATION_CONFIG)
    braw,bwarn=shared.load_normalized(Path(before_path));araw,awarn=shared.load_normalized(Path(after_path))
    bd=shared.build_duplicate_groups(braw,bcfg);ad=shared.build_duplicate_groups(araw,bcfg)
    bs=shared.group_representative_elems(braw,bd);ats=shared.group_representative_elems(araw,ad)
    bi=Image.open(before_image).convert('RGB');ai=Image.open(after_image).convert('RGB')
    anchors=shared.build_ocr_anchors(bs,ats,bcfg)
    anchors['container_candidates']=shared.match_containers(bs,ats,anchors,bcfg)
    anchors['textures']={}
    for side,es,im in [('before',bs,bi),('after',ats,ai)]:
        descriptors={}
        for e in es:
            box=intersect(e['bbox'],[0,0,im.width,im.height])
            if area(box)>0:
                ar=np.asarray(im.crop(tuple(box)).convert('L').resize((32,12)),dtype=float)
                descriptors[e['id']]=(ar-np.median(ar)).tolist()
        anchors['textures'][side]=descriptors
    # Exact OCR is a soft bonus/prior only. Every representative enters assignment.
    bh=shared.precompute_crop_histograms(bs,bi,32);ah=shared.precompute_crop_histograms(ats,ai,32)
    result=shared.solve_assignment(bs,ats,set(),set(),math.hypot(bi.width,bi.height),anchors,cfg,bh,ah,cost_builder=content_cost)
    bl={e['id']:e for e in bs};al={e['id']:e for e in ats}
    for record in result['matched_pairs']+result['uncertain_pairs']:
        b,a=bl[record['before_id']],al[record['after_id']]
        record.update(before_source=b['source'],after_source=a['source'],before_members=b['group_members'],after_members=a['group_members'],
                      delta=shared.calc_delta(b,a,math.hypot(bi.width,bi.height)),content=classify(b,a,bi,ai,braw,araw,cfg))
    predictions=[dict(r,prediction_index=i) for i,r in enumerate(r for r in result['matched_pairs'] if r['content']['state']=='changed')]
    result.update(config=cfg,shared_duplicate_anchor_config=bcfg,uses_groundtruth=False,before_detections=braw,after_detections=araw,
                  duplicates={'before':bd,'after':ad},anchors=anchors,predictions=predictions,groups=groups(predictions,cfg),warnings=bwarn+awarn,
                  image_size=[bi.width,bi.height])
    # Assert one-to-one AND full accounting after duplicate normalization.
    for side,es in [('before',bs),('after',ats)]:
        used=[r[side+'_id'] for r in result['matched_pairs']+result['uncertain_pairs']]
        un=[e['id'] for e in result['unmatched_'+side]]
        assert len(used)==len(set(used)) and set(used).isdisjoint(un)
        assert set(used+un)=={e['id'] for e in es}
    return result

def support_metrics(boxes,gt):
    inter=union_area([intersect(b,gt) for b in boxes]);sup=union_area(boxes)
    return {'gt_fraction':inter/max(area(gt),1e-9),'support_fraction':inter/max(sup,1e-9),
            'iou':inter/max(area(gt)+sup-inter,1e-9)}

def evaluate(result,gt_data,cfg=None):
    """Many-to-many GT coverage using actual endpoints, never the motion envelope."""
    cfg=cfg or CONFIG;pred=result['predictions'];targets=[];used=set()
    supports=[{'kind':'single','prediction_indices':[i],'boxes':[p['before_bbox'],p['after_bbox']]} for i,p in enumerate(pred)]
    supports += [{'kind':'group','group_id':g['group_id'],'prediction_indices':g['member_prediction_indices'],'boxes':g['support_boxes']} for g in result['groups']]
    for gi,g in enumerate(gt_data['bboxes']):
        if 'content' not in g['type']:continue
        box=[g['x'],g['y'],g['x']+g['w'],g['y']+g['h']];links=[]
        for s in supports:
            metrics=support_metrics(s['boxes'],box)
            if metrics['gt_fraction']>=cfg['coverage_gt_fraction'] and metrics['support_fraction']>=cfg['coverage_support_fraction']:
                links.append({k:v for k,v in s.items() if k!='boxes'}|metrics);used.update(s['prediction_indices'])
        # Localization diagnostics only; not a correctness label.
        diagnostics=[]
        if not links:
            overlap=lambda r:any(area(intersect(r[side],box))>0 for side in ['before_bbox','after_bbox'])
            nearby=[r for r in result['matched_pairs'] if overlap(r)]
            if not any(overlap({'before_bbox':e['bbox'],'after_bbox':e['bbox']}) for e in result['before_detections']+result['after_detections']):
                diagnostics.append('detector_no_overlap')
            if any(overlap(r) for r in result['uncertain_pairs']):diagnostics.append('correspondence_uncertain')
            if any(area(intersect(e['bbox'],box))>0 for side in ['before','after'] for e in result['unmatched_'+side]):diagnostics.append('correspondence_unmatched_or_segmentation')
            if any(r['content']['state']!='changed' for r in nearby):diagnostics.append('change_evidence_insufficient_or_gt_unchanged')
            if any(overlap(r) for r in pred):diagnostics.append('prediction_region_or_coverage_threshold')
            if not diagnostics:diagnostics.append('correspondence_or_detector_localization_review')
        targets.append({'original_gt_index':gi,'gt':g,'covered':bool(links),'links':links,'diagnostic_hypotheses':diagnostics,'human_cause':None})
    return {'gt_count':len(targets),'covered':sum(t['covered'] for t in targets),
            'ocr_supported_covered':sum(any(all(pred[i].get('content',{}).get('kind')=='ocr_and_visual' for i in link['prediction_indices']) for link in t['links']) for t in targets),'pass':all(t['covered'] for t in targets),
            'predictions':len(pred),'unverified_predictions':len(pred)-len(used),'gt_coverage':targets,
            'protocol':'endpoint_union_gt>=0.50_AND_support>=0.25; singleton or local group; many-to-many',
            'not_precision_recall_f1':True}

def run(out,split):
    manifest=read(out/'data_manifest.json');cfg=read(out/'config.json')
    if split=='all':
        frozen=read(out/'frozen_config.json')
        if frozen['config_sha256']!=sha(out/'config.json') or frozen['code_sha256']!=sha(__file__) or frozen['shared_code_sha256']!=sha(ROOT/'shared/matching.py'):
            raise ValueError('Config/code differs from development freeze; rerun dev and review')
    selected=[s for s in manifest['samples'] if split=='all' or s['split']=='dev']
    write(out/f'{split}_status.json',{'status':'inference_without_gt'})
    artifacts=[]
    for s in selected:
        pid=s['pair_id'];src=s['sources'];dest=out/pid
        r=infer(src['before'],src['after'],src['before_image'],src['after_image'],cfg)
        write(dest/'inference.json',r)
        artifacts.append({'pair_id':pid,'sha256':sha(dest/'inference.json')})
        for side in ['before','after']:shutil.copy2(src[side+'_image'],dest/f'{side}.png')
        print(pid, 'matched',len(r['matched_pairs']),'uncertain',len(r['uncertain_pairs']),'predictions',len(r['predictions']),flush=True)
    write(out/f'{split}_prediction_manifest.json',{'phase':'all_predictions_frozen_before_scoring','artifacts':artifacts})
    rows=[]
    for s in selected:
        dest=out/s['pair_id'];r=read(dest/'inference.json')
        ev=evaluate(r,read(s['sources']['groundtruth']),cfg);write(dest/'evaluation.json',ev)
        rows.append({'pair_id':s['pair_id'],'split':s['split'],**{k:ev[k] for k in ['gt_count','covered','ocr_supported_covered','pass','predictions','unverified_predictions']},
                     'uncertain_correspondences':len(r['uncertain_pairs']),
                     'unmatched_before':len(r['unmatched_before']),'unmatched_after':len(r['unmatched_after'])})
    for a in artifacts:assert sha(out/a['pair_id']/'inference.json')==a['sha256']
    for source in manifest['inputs']:assert sha(source['path'])==source['sha256']
    with (out/f'{split}_evaluation.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for name in ['dev','remaining','all']:
        subset=[r for r in rows if name=='all' or r['split']==name]
        summary[name]={'samples':len(subset),**{k:sum(r[k] for r in subset) for k in ['gt_count','covered','ocr_supported_covered','predictions','unverified_predictions']},'pass_samples':sum(r['pass'] for r in subset)}
        summary[name]['coverage']=summary[name]['covered']/max(summary[name]['gt_count'],1)
    write(out/f'{split}_summary.json',summary)
    write(out/f'{split}_failures.json',[{'pair_id':r['pair_id'],'missed':[g for g in read(out/r['pair_id']/'evaluation.json')['gt_coverage'] if not g['covered']]} for r in rows if not r['pass']])
    write(out/f'{split}_status.json',{'status':'complete','pair_ids':[s['pair_id'] for s in selected],
                                    'source_hashes_verified':True,'prediction_hashes_verified':True,'config_sha256':sha(out/'config.json'),
                                    'code_sha256':sha(__file__),'shared_code_sha256':sha(ROOT/'shared/matching.py')})
    print(json.dumps(summary,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['survey','run','freeze','report'])
    parser.add_argument('--split',choices=['dev','all'],default='dev')
    parser.add_argument('--output-root',type=Path,default=ROOT/'content/results/organized_v1')
    args=parser.parse_args();out=args.output_root.resolve()
    if not out.is_relative_to(ROOT/'content/results') or out==ROOT/'content/results':
        parser.error('output-root must be a child of content/results; sources/geometry protected')
    if args.command=='survey':
        out.mkdir(parents=True,exist_ok=True)
        inv=inventory();write(out/'data_manifest.json',inv)
        if not (out/'config.json').exists():write(out/'config.json',CONFIG)
        write(out/'development_ids.json',{'rule':'first five numeric content IDs; chosen before inference','pair_ids':DEV_IDS})
        print(json.dumps({k:v for k,v in inv.items() if k not in ['samples','inputs']},indent=2))
    elif args.command=='run':run(out,args.split)
    elif args.command=='freeze':
        status=read(out/'dev_status.json');assert status['status']=='complete'
        assert (out/'development_review.md').exists(),'Record development review before freeze'
        for path,key in [(out/'config.json','config_sha256'),(Path(__file__),'code_sha256'),(ROOT/'shared/matching.py','shared_code_sha256')]:
            assert status[key]==sha(path),'Development run is stale'
        write(out/'frozen_config.json',{'config_sha256':sha(out/'config.json'),'code_sha256':sha(__file__),
              'shared_code_sha256':sha(ROOT/'shared/matching.py'),'development_ids':DEV_IDS,
              'review_record':'development_review.md','review_sha256':sha(out/'development_review.md'),'remaining_metadata_and_detections_seen_in_survey':True,
              'remaining_image_reviewed_before_freeze':read(out/'review_context.json').get('remaining_image_reviewed',False) if (out/'review_context.json').exists() else False,
              'independent_test':False,'note':'Observed cohort; see review_record and review_context.json for prior exposure. No independent test claim.'})
    elif args.command=='report':
        from content.scripts.build_content_report import build
        build(out)

if __name__=='__main__':main()
