"""Review every missed GT; independent motion groups remain unchanged."""
import gzip
import json
import os
from collections import Counter
from pathlib import Path
from PIL import Image
from geometry import pipeline
from geometry.change_evidence import contained_group_metrics
from geometry.scripts.detect_dataset2 import ROOT,save
from geometry.scripts.analyze_dataset2_failures import metrics,passes
OUT=ROOT/'geometry/results/coarse_gt_event_audit_v1'

def main():
    failures=json.loads((OUT/'failure_audit.json').read_text(encoding='utf-8'));by_sample={}
    for r in failures:by_sample.setdefault(r['pair_id'],[]).append(r)
    views=[];secondary=[]
    for pid,targets in by_sample.items():
        first=targets[0]['dataset']=='first';source=ROOT/'geometry/results' if first else ROOT/'geometry/results/dataset2_from502_detection_v1'
        r=json.loads((source/pid/'matching_result_full.json').read_text(encoding='utf-8')) if first else json.loads(gzip.open(source/pid/'matching_result_full.json.gz','rt').read());evidence=json.loads((OUT/pid/'visual_evidence_and_groups.json').read_text(encoding='utf-8'));changed=[m for m in r['matched_pairs'] if pipeline.classify_change(m['delta'],'edges8')];uncertain=[m for m in r['uncertain_pairs'] if pipeline.classify_change(m['delta'],'edges8')];unchanged=[m for m in r['matched_pairs'] if not pipeline.classify_change(m['delta'],'edges8')]
        raw={}
        for side,stage in [('before','baseline'),('after','inject')]:
            if first:path=ROOT/'geometry/pairs'/pid/f'{side}.json'
            else:path=next((ROOT/'dataset2_detection'/stage/'json').glob(pid+'_*_full_hierarchy.json'))
            raw[side]=json.loads(path.read_text(encoding='utf-8'))['elements']
        for t in targets:
            gt=t['gt'];gb=[gt['x'],gt['y'],gt['x']+gt['w'],gt['y']+gt['h']];candidates=[]
            for kind,records in [('single',changed),('uncertain',uncertain),('below_classifier',unchanged)]:
                for i,m in enumerate(records):
                    region=pipeline.prediction_envelope(m['before_bbox'],m['after_bbox']);value=contained_group_metrics([region],gb)
                    if value['prediction_containment']<=0:continue
                    candidates.append({'kind':kind,'id':f'{kind}_{i}','prediction_indices':[i],'before':[m['before_bbox']],'after':[m['after_bbox']],'regions':[region],'before_ids':[m['before_id']],'after_ids':[m['after_id']],'visual_evidence':evidence['visual_features'][i] if kind=='single' else {'counted_in_event_metric':False,'reason':m.get('uncertain_reason'),'delta':m['delta']},**value,'ccr':value['gt_fraction']})
            for g in evidence['verified_motion_groups']:
                value=contained_group_metrics(g['member_regions'],gb)
                if value['prediction_containment']<=0:continue
                candidates.append({'kind':'group','id':g['group_id'],'prediction_indices':g['original_prediction_indices'],'before':[m['before_bbox'] for m in g['members']],'after':[m['after_bbox'] for m in g['members']],'regions':g['member_regions'],'before_ids':[m['before_id'] for m in g['members']],'after_ids':[m['after_id'] for m in g['members']],'visual_evidence':[m['visual_evidence'] for m in g['members']],**value,'ccr':value['gt_fraction']})
            candidates.sort(key=lambda c:({'group':4,'single':3,'uncertain':2,'below_classifier':1}[c['kind']],c['prediction_containment'],c['iou']),reverse=True)
            if not candidates:candidates=[{'kind':'no_candidate','id':'none','prediction_indices':[],'before':[],'after':[],'regions':[],'before_ids':[],'after_ids':[],'visual_evidence':{'note':'No overlapping changed, uncertain-changed or below-classifier correspondence.'},'prediction_containment':0,'gt_fraction':0,'iou':0,'ccr':0}]
            sizes={}
            for side in ('before','after'):
                with Image.open(r['image_paths'][side]) as im:sizes[side]=list(im.size)
            views.append({'case_id':f"{pid}-GT{t['original_gt_index']+1}",'pair_id':pid,'dataset':t['dataset'],'gt':gt,'original_gt_index':t['original_gt_index'],'diagnostic':t['diagnostic'],'event_covered':t['event_covered_experimental'],'candidates':candidates,'candidate_count':len(candidates),'before':os.path.relpath(r['image_paths']['before'],OUT),'after':os.path.relpath(r['image_paths']['after'],OUT),'image_sizes':sizes})
            if t['diagnostic']=='no_changed_matched_prediction_intersects_gt':
                uv=[c for c in candidates if c['kind']=='uncertain' and passes(c)];mv=[c for c in candidates if c['kind']=='below_classifier' and passes(c)]
                raw_stats={s:{'best_single':max((metrics([d['bbox']],gb) for d in dets),key=lambda q:max(q['iou']/.3,q['ccr']/.5),default=None),'union_all':metrics([d['bbox'] for d in dets],gb)} for s,dets in raw.items()}
                if uv:reason='uncertain_changed_overlap_passes'
                elif mv:reason='matched_below_classifier_overlap_passes'
                elif any(passes(v['union_all']) or (v['best_single'] and passes(v['best_single'])) for v in raw_stats.values()):reason='raw_detection_coverage_available_but_no_changed_correspondence'
                else:reason='raw_detection_coverage_insufficient'
                secondary.append({'pair_id':pid,'original_gt_index':t['original_gt_index'],'gt':gt,'condition':reason,'raw_detection_stats':raw_stats,'uncertain_candidate_count':len(uv),'below_classifier_candidate_count':len(mv),'note':'Overlap diagnostics do not establish semantic identity; raw union is an upper bound, not a valid prediction.'})
    save(OUT/'viewer_cases.json',views);save(OUT/'no_changed_prediction_79_trace.json',{'counts':dict(Counter(r['condition'] for r in secondary)),'cases':secondary})
    template=(ROOT/'geometry/scripts/all_low_overlap_review.html').read_text(encoding='utf-8').replace('63 mẫu · 79 GT thiếu overlap','Truy vết toàn bộ 158 GT missed').replace('Tập đầu · 25 mẫu / 33 GT','Tập đầu · 33 mẫu / 50 GT missed').replace('Tập sau · 38 mẫu / 46 GT','Tập sau · 60 mẫu / 108 GT missed')
    template=template.replace('Chỉ hiện GT bị bỏ sót có prediction/nhóm thay đổi giao GT nhưng IoU &lt; 0,30 và CCR &lt; 0,50.','Có cả trường hợp chưa có prediction thay đổi. Metric event thử nghiệm và điểm localization được tách biệt; không xác nhận GT sai.')
    template=template.replace("p.kind==='group'?'Nhóm':'Prediction'","p.kind==='group'?'Nhóm có bằng chứng ảnh':p.kind==='uncertain'?'Uncertain — không chấm':p.kind==='below_classifier'?'Dưới ngưỡng — không chấm':p.kind==='no_candidate'?'Không có cặp giao GT':'Prediction'")
    template=template.replace(' · FAIL`'," · chứa ${(100*p.prediction_containment).toFixed(1)}% · GT ${c.event_covered?'có event hit thử nghiệm':'chưa đủ event evidence'}`").replace('delta:p.delta,','delta:p.delta,diagnostic:c.diagnostic,prediction_containment:p.prediction_containment,visual_evidence:p.visual_evidence,')
    start=template.find('<p>Ảnh tổng quan before/after');end=template.find('</p>',start)
    if start>=0:template=template[:start]+template[end+4:]
    (OUT/'index.html').write_text(template.replace('/*CASES*/[]',json.dumps(views,ensure_ascii=False).replace('<','\\u003c')), encoding='utf-8')
    marker='\n## Truy vết 79 GT không có matched-changed giao vùng'
    p=OUT/'REPORT.md';p.write_text(p.read_text(encoding='utf-8').split(marker)[0]+marker+'\n\n```json\n'+json.dumps(dict(Counter(r['condition'] for r in secondary)),indent=2)+'\n```\n\nCác điều kiện này không tự xác nhận correspondence đúng; có detection bao GT không chứng minh detector bắt đúng target. `index.html` hiện đầy đủ 158 GT missed, gồm cả uncertain và matched dưới ngưỡng để quan sát (không tính chúng vào metric event).\n', encoding='utf-8')
    print(json.dumps({'viewer_samples':len(by_sample),'viewer_gt':len(views),'no_changed_79_conditions':dict(Counter(r['condition'] for r in secondary))},indent=2))
if __name__=='__main__':main()
