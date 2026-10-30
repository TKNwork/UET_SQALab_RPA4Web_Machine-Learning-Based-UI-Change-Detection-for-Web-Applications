"""Trace localization failures and separately test visually supported coarse-GT events."""
import gzip
import json
import os
from pathlib import Path
from collections import Counter
from PIL import Image
from geometry import pipeline
from geometry.change_evidence import CONFIG,motion_features,event_links,contained_group_metrics
from geometry.hierarchical_groups import build_hierarchical
from geometry.prediction_groups import GROUP_CONFIG
from geometry.scripts.detect_dataset2 import ROOT,save,sha
from shared.regions import area,intersect
OUT=ROOT/'geometry/results/coarse_gt_event_audit_v1'
GT=ROOT.parent/'dataset1/thesis-dataset-main/groundtruth'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(source,pid,first):return read(source/pid/'matching_result_full.json') if first else json.loads(gzip.open(source/pid/'matching_result_full.json.gz','rt').read())
def main():
    OUT.mkdir(exist_ok=True);save(OUT/'frozen_config.json',{'motion_evidence':CONFIG,'note':'All data already observed; exploratory event metric, not replacement localization benchmark or proven precision.'});records=[];protected=[]
    first=read(ROOT/'geometry/results/matching_manifest.json')['pair_ids'];second=[r['pair_id'] for r in read(ROOT/'geometry/results/dataset2_same_gt_protocol_v1/data_manifest.json')['samples'] if r['geometry_count']]
    # Freeze motion-qualified groups without loading GT coordinates.
    for ds,ids,source,base in [('first',first,ROOT/'geometry/results',ROOT/'geometry/results/group_coverage_v1'),('second',second,ROOT/'geometry/results/dataset2_from502_detection_v1',ROOT/'geometry/results/dataset2_same_gt_protocol_v1')]:
        for pid in ids:
            r=load(source,pid,ds=='first');selected=[m for m in r['matched_pairs'] if pipeline.classify_change(m['delta'],'edges8')]
            for p in [source/pid/('matching_result_full.json' if ds=='first' else 'matching_result_full.json.gz'),base/pid/'eval_edges8_union_support.json',GT/f'{int(pid):03d}.json']:protected.append({'path':str(p),'sha256':sha(p)})
            with Image.open(r['image_paths']['before']).convert('RGB') as b,Image.open(r['image_paths']['after']).convert('RGB') as a:
                features=[motion_features(b,a,m) for m in selected];indices=[i for i,q in enumerate(features) if q['strong']];strong=[selected[i] for i in indices]
                groups=build_hierarchical(strong,r['matched_pairs'],b.size,{'config':GROUP_CONFIG,'groups':[],'selected_predictions':len(strong),'uses_groundtruth':False},pixel_scorer=None)['groups']
            for g in groups:
                g['original_prediction_indices']=[indices[i] for i in g['member_prediction_indices']]
                for member in g['members']:member['visual_evidence']=features[indices[member['prediction_index']]]
            evidence={'selected_prediction_count':len(selected),'visual_features':features,'strong_prediction_indices':indices,'verified_motion_groups':groups,'uses_groundtruth':False}
            save(OUT/pid/'visual_evidence_and_groups.json',evidence);records.append({'pair_id':pid,'dataset':ds,'source':str(source),'baseline':str(base)})
    artifacts=[{'path':str(OUT/r['pair_id']/'visual_evidence_and_groups.json'),'sha256':sha(OUT/r['pair_id']/'visual_evidence_and_groups.json')} for r in records]
    save(OUT/'frozen_prediction_manifest.json',{'phase':'all_motion_evidence_and_groups_frozen_before_GT_scoring','artifacts':artifacts,'config_sha256':sha(OUT/'frozen_config.json'),'model_code_sha256':sha(ROOT/'geometry/change_evidence.py'),'group_code_sha256':sha(ROOT/'geometry/hierarchical_groups.py')})
    results=[];failures=[];view=[]
    for row in records:
        pid=row['pair_id'];r=load(Path(row['source']),pid,row['dataset']=='first');selected=[m for m in r['matched_pairs'] if pipeline.classify_change(m['delta'],'edges8')];evidence=read(OUT/pid/'visual_evidence_and_groups.json');base=read(Path(row['baseline'])/pid/'eval_edges8_union_support.json');raw=read(GT/f'{int(pid):03d}.json')['bboxes'];coverage=[];sample_cases=[]
        for cov in base['gt_coverage']:
            gt=raw[cov['original_gt_index']];gb=[gt['x'],gt['y'],gt['x']+gt['w'],gt['y']+gt['h']];links=event_links(evidence['verified_motion_groups'],gb);contained=[];overlap=[]
            for i,m in enumerate(selected):
                region=pipeline.prediction_envelope(m['before_bbox'],m['after_bbox']);value=contained_group_metrics([region],gb)
                if area(intersect(region,gb))>0:overlap.append(i)
                if value['prediction_containment']>=CONFIG['min_group_containment']:contained.append({'prediction_index':i,'visual_evidence':evidence['visual_features'][i],**value})
            strong_contained=[x for x in contained if x['visual_evidence']['strong']]
            new_covered=cov['covered'] or bool(links)
            target={'original_gt_index':cov['original_gt_index'],'gt':gt,'localization_covered':cov['covered'],'event_covered_experimental':new_covered,'visually_supported_group_links':links,'contained_changed':contained,'strong_contained_count':len(strong_contained)}
            if not cov['covered']:
                if links:reason='coarse_gt_metric_mismatch_supported_by_visual_motion_group'
                elif strong_contained:reason='single_or_ungrouped_visual_motion_inside_gt_needs_semantic_review'
                elif contained:reason='contained_prediction_but_image_motion_not_verified'
                elif overlap:reason='changed_prediction_intersects_gt_but_not_contained'
                else:reason='no_changed_matched_prediction_intersects_gt'
                target['diagnostic']=reason;target['human_gt_error']=None;failures.append(dict(pair_id=pid,dataset=row['dataset'],**target));sample_cases.append(target)
            coverage.append(target)
        ev={'protocol':'baseline_localization_OR_visually_supported_contained_motion_group_experimental_v1','gt_count':base['gt_count'],'localization_covered':base['covered'],'event_covered_experimental':sum(c['event_covered_experimental'] for c in coverage),'event_pass_experimental':all(c['event_covered_experimental'] for c in coverage),'baseline_pass':base['pass'],'gt_coverage':coverage,'not_proven_correspondence_accuracy':True}
        save(OUT/pid/'evaluation.json',ev);results.append({'pair_id':pid,'dataset':row['dataset'],**{k:ev[k] for k in ('gt_count','localization_covered','event_covered_experimental','event_pass_experimental','baseline_pass')}})
        if sample_cases:
            candidates=[]
            # Keep every contained singleton visible for audit, including rejected evidence.
            for i,m in enumerate(selected):
                candidates.append({'kind':'single','id':f'prediction_{i}','prediction_indices':[i],'before':[m['before_bbox']],'after':[m['after_bbox']],'regions':[pipeline.prediction_envelope(m['before_bbox'],m['after_bbox'])],'before_ids':[m['before_id']],'after_ids':[m['after_id']],'visual_evidence':evidence['visual_features'][i]})
            for g in evidence['verified_motion_groups']:
                candidates.append({'kind':'group','id':g['group_id'],'prediction_indices':g['original_prediction_indices'],'before':[m['before_bbox'] for m in g['members']],'after':[m['after_bbox'] for m in g['members']],'regions':g['member_regions'],'before_ids':[m['before_id'] for m in g['members']],'after_ids':[m['after_id'] for m in g['members']],'visual_evidence':[m['visual_evidence'] for m in g['members']]})
            for t in sample_cases:
                g=t['gt'];gb=[g['x'],g['y'],g['x']+g['w'],g['y']+g['h']];cs=[]
                for c in candidates:
                    value=contained_group_metrics(c['regions'],gb)
                    if value['prediction_containment']>0:cs.append(dict(c,**value,ccr=value['gt_fraction']))
                cs.sort(key=lambda c:(c['kind']=='group' and c['prediction_containment']>=.9,c['prediction_containment'],c['iou']),reverse=True)
                if cs:view.append({'case_id':f"{pid}-GT{t['original_gt_index']+1}",'pair_id':pid,'dataset':row['dataset'],'gt':g,'original_gt_index':t['original_gt_index'],'diagnostic':t['diagnostic'],'event_covered':t['event_covered_experimental'],'candidates':cs,'candidate_count':len(cs),'before':os.path.relpath(r['image_paths']['before'],OUT),'after':os.path.relpath(r['image_paths']['after'],OUT),'image_sizes':{s:list(Image.open(r['image_paths'][s]).size) for s in ('before','after')}})
    summary={}
    for ds in ('first','second','all'):
        rows=[r for r in results if ds=='all' or r['dataset']==ds];summary[ds]={k:sum(r[k] for r in rows) for k in ('gt_count','localization_covered','event_covered_experimental','event_pass_experimental','baseline_pass')};summary[ds]['samples']=len(rows)
    summary['failure_diagnostics']=dict(Counter(r['diagnostic'] for r in failures));summary['gt_errors_confirmed']=0;summary['note']='No GT labels were changed. Event coverage is a distinct exploratory metric; no claim that detector found every change or that GT is wrong.'
    assert len(failures)==158 and summary['all']['gt_count']==691 and summary['all']['localization_covered']==533
    for e in protected+artifacts:assert sha(Path(e['path']))==e['sha256']
    save(OUT/'summary.json',summary);save(OUT/'all_results.json',results);save(OUT/'failure_audit.json',failures);save(OUT/'validation.json',{'geometry_samples':296,'original_gt':691,'baseline_gt_coverage_unchanged':533,'all_158_missed_gt_audited':True,'motion_features_and_groups_generated_without_gt':True,'all_original_inputs_and_frozen_artifacts_preserved':True,'human_gt_errors_confirmed':0})
    template=(ROOT/'geometry/scripts/all_low_overlap_review.html').read_text(encoding='utf-8').replace('63 mẫu · 79 GT thiếu overlap','Truy vết GT và bằng chứng chuyển động').replace('Tập đầu · 25 mẫu / 33 GT','Tập đầu').replace('Tập sau · 38 mẫu / 46 GT','Tập sau')
    template=template.replace('Chỉ hiện GT bị bỏ sót có prediction/nhóm thay đổi giao GT nhưng IoU &lt; 0,30 và CCR &lt; 0,50.','GT missed của baseline có prediction giao vùng: xem bằng chứng ảnh và quan hệ chứa; không xác nhận GT sai.')
    template=template.replace(' · FAIL`'," · chứa ${(100*p.prediction_containment).toFixed(1)}% · ${c.event_covered?'Event hit thử nghiệm':'Chưa đủ bằng chứng'}`").replace('delta:p.delta,','delta:p.delta,diagnostic:c.diagnostic,prediction_containment:p.prediction_containment,visual_evidence:p.visual_evidence,')
    start=template.find('<p>Ảnh tổng quan before/after');end=template.find('</p>',start)
    if start>=0:template=template[:start]+template[end+4:]
    (OUT/'index.html').write_text(template.replace('/*CASES*/[]',json.dumps(view,ensure_ascii=False).replace('<','\\u003c')), encoding='utf-8')
    lines=['# Audit GT thô và event coverage — toàn bộ geometry','','Giữ nguyên 691 GT, prediction, matching, classifier và điểm baseline. Chưa xác nhận GT nào sai; đây là audit và metric thử nghiệm, không sửa nhãn để tăng điểm.','','## Bằng chứng trước khi chấm','','Với mỗi matched đã vượt edges8, so crop before/after sau căn chỉnh bbox và so với crop tại vị trí đứng yên. Chỉ giữ crop đủ texture, tương đồng sau căn chỉnh ≥0.80, lợi ích căn chỉnh ≥0.02, pixel đổi >24 trên ≥1%, bbox đủ trong ảnh, và correspondence có OCR exact hoặc global stability đạt. Sau đó dựng nhóm ≥2 thành viên mạnh với chuyển động tương thích, độc lập GT.','','## Cách chấm bổ sung','','GT chỉ cần chứa ≥90% diện tích union của một nhóm chuyển động mạnh (không yêu cầu prediction phủ 50% GT). Đây là **event-hit coverage cho GT thô**, không phải localization IoU/CCR hoặc chứng minh đủ mọi phần tử trong GT. Nhóm và ảnh chứng minh có thay đổi ở vùng GT, chưa chứng minh danh tính target selector là đúng. Điểm baseline được báo riêng; không tự thay protocol chính.','','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','','`failure_audit.json`: từng GT với lý do, ứng viên, feature ảnh và link nhóm. `index.html`: so ảnh/crop và bằng chứng từng ứng viên. Không có bằng chứng cũng được ghi lại, không quy hết lỗi cho GT.','','Dữ liệu đã được quan sát; đây là nghiên cứu development, không phải held-out validation. Chưa có precision trên vùng không thay đổi.','','Tái lập: `python3 -m geometry.scripts.audit_coarse_gt_events`.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    from geometry.scripts.review_coarse_gt_audit import main as review
    review()
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
