"""Freeze GT-independent hierarchy/layout proposals, then score all 296 geometry samples."""
import gzip
import json
import math
import os
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from geometry import pipeline
from geometry.hierarchical_groups import CONFIG,build_hierarchical
from shared.regions import area,intersect
from geometry.scripts.analyze_dataset2_failures import metrics,passes
from geometry.scripts.detect_dataset2 import ROOT,save,sha
OUT=ROOT/'geometry/results/hierarchical_geometry_v1'
GT=ROOT.parent/'dataset1/thesis-dataset-main/groundtruth'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load_r(source,pid,first):
    p=source/pid/('matching_result_full.json' if first else 'matching_result_full.json.gz')
    return json.loads(p.read_text(encoding='utf-8')) if first else json.loads(gzip.open(p,'rt').read())
def pixel_scorer(before_path,after_path):
    b=np.asarray(Image.open(before_path).convert('RGB'),dtype=np.int16);a=np.asarray(Image.open(after_path).convert('RGB'),dtype=np.int16);h=min(b.shape[0],a.shape[0]);w=min(b.shape[1],a.shape[1]);mask=(np.max(np.abs(b[:h,:w]-a[:h,:w]),axis=2)>CONFIG['pixel_channel_difference'])
    def score(region):
        x1=max(0,math.floor(region[0]));y1=max(0,math.floor(region[1]));x2=min(w,math.ceil(region[2]));y2=min(h,math.ceil(region[3]));view=mask[y1:max(y1,y2),x1:max(x1,x2)]
        return {'change_fraction':float(view.mean()) if view.size else 0.,'visible_fraction':area(intersect(region,[0,0,w,h]))/max(area(region),1),'channel_difference_threshold':CONFIG['pixel_channel_difference']}
    return score

def aggregate(rows,key):
    return {'samples':len(rows),'gt_count':sum(r['gt_count'] for r in rows),'covered':sum(r[key]['covered'] for r in rows),'pass_samples':sum(r[key]['pass'] for r in rows),'fail_samples':sum(not r[key]['pass'] for r in rows)}

def main():
    OUT.mkdir(exist_ok=True);save(OUT/'frozen_config.json',{'hierarchy':CONFIG,'classifier':'edges8','thresholds':{'iou':.3,'ccr':.5},'note':'Frozen before full scoring. Development uses already observed 79 low-overlap cases; no held-out performance claim.'})
    oldids=read(ROOT/'geometry/results/matching_manifest.json')['pair_ids'];newmanifest=read(ROOT/'geometry/results/dataset2_same_gt_protocol_v1/data_manifest.json');newids=[r['pair_id'] for r in newmanifest['samples'] if r['geometry_count']]
    records=[];protected=[]
    # All proposal generation is completed and persisted before reading GT for scoring.
    for dataset,ids,source,baseline in [('first',oldids,ROOT/'geometry/results',ROOT/'geometry/results/group_coverage_v1'),('second',newids,ROOT/'geometry/results/dataset2_from502_detection_v1',ROOT/'geometry/results/dataset2_same_gt_protocol_v1')]:
        for pid in ids:
            r=load_r(source,pid,dataset=='first');groups_path=(baseline if dataset=='first' else source)/pid/'groups_edges8.json';basegroups=read(groups_path)
            p=source/pid/('matching_result_full.json' if dataset=='first' else 'matching_result_full.json.gz')
            for path in (p,groups_path,baseline/pid/'eval_edges8_union_support.json',GT/f'{int(pid):03d}.json'):protected.append({'path':str(path),'sha256':sha(path)})
            with Image.open(r['image_paths']['before']) as im:size=im.size
            selected=[m for m in r['matched_pairs'] if pipeline.classify_change(m['delta'],'edges8')]
            new=build_hierarchical(selected,r['matched_pairs'],size,basegroups,pixel_scorer(r['image_paths']['before'],r['image_paths']['after']))
            save(OUT/pid/'groups_and_layout_proposals.json',new)
            records.append({'pair_id':pid,'dataset':dataset,'source':str(source),'baseline':str(baseline)})
    save(OUT/'proposal_manifest.json',{'phase':'all_groups_and_regions_frozen_before_scoring','pair_ids':[r['pair_id'] for r in records],'code_sha256':sha(ROOT/'geometry/hierarchical_groups.py'),'config_sha256':sha(OUT/'frozen_config.json'),'inputs':protected})
    results=[];newlinks=[]
    for row in records:
        pid=row['pair_id'];r=load_r(Path(row['source']),pid,row['dataset']=='first');groups=read(OUT/pid/'groups_and_layout_proposals.json');baseline=read(Path(row['baseline'])/pid/'eval_edges8_union_support.json');gp=GT/f'{int(pid):03d}.json'
        union=pipeline.evaluate_grouped(r['matched_pairs'],gp,'edges8',groups,'union_support');assert union['covered']>=baseline['covered']
        gts=[(i,g) for i,g in enumerate(read(gp)['bboxes']) if 'geometry' in g.get('type',[])];covered={c['gt_index'] for c in union['gt_coverage'] if c['covered']};links=[]
        for proposal in groups['layout_proposals']:
            for gi,(orig,g) in enumerate(gts):
                gb=[g['x'],g['y'],g['x']+g['w'],g['y']+g['h']];value=metrics([proposal['region']],gb)
                if passes(value):links.append({'gt_index':gi,'original_gt_index':orig,'proposal_id':proposal['proposal_id'],'evidence_type':proposal['evidence_type'],'iou':value['iou'],'ccr':value['ccr'],'actual_member_union':metrics(proposal['member_regions'],gb),'needs_human_review':True})
        allcovered=covered|{l['gt_index'] for l in links};experimental={'gt_count':len(gts),'covered':len(allcovered),'pass':len(allcovered)==len(gts),'protocol':'union_support_plus_gt_independent_inferred_layout_extent_experimental','layout_links':links,'missed_gt':sorted(set(range(len(gts)))-allcovered),'gt_coverage':[dict(c,covered=c['gt_index'] in allcovered,layout_proposal_ids=[l['proposal_id'] for l in links if l['gt_index']==c['gt_index']]) for c in union['gt_coverage']]}
        save(OUT/pid/'eval_hierarchy_union.json',union);save(OUT/pid/'eval_layout_experimental.json',experimental)
        new_gt=sorted(allcovered-{c['gt_index'] for c in baseline['gt_coverage'] if c['covered']})
        for gi in new_gt:
            newlinks.append({'pair_id':pid,'dataset':row['dataset'],'gt_index':gi,'original_gt_index':gts[gi][0],'rescued_by_union':gi in covered,'layout_links':[l for l in links if l['gt_index']==gi],'human_correct':None})
        results.append({'pair_id':pid,'dataset':row['dataset'],'gt_count':len(gts),'baseline':{'covered':baseline['covered'],'pass':baseline['pass']},'hierarchy_union':{'covered':union['covered'],'pass':union['pass']},'layout_experimental':{'covered':experimental['covered'],'pass':experimental['pass']},'new_groups':len(groups['groups'])-baseline['group_count'],'layout_proposals':len(groups['layout_proposals'])})
    summary={}
    for dataset in ('first','second','all'):
        rows=[r for r in results if dataset=='all' or r['dataset']==dataset];summary[dataset]={mode:aggregate(rows,mode) for mode in ('baseline','hierarchy_union','layout_experimental')}
        for v in summary[dataset].values():v['coverage']=v['covered']/max(v['gt_count'],1)
    assert summary['all']['baseline']['gt_count']==691 and summary['all']['baseline']['covered']==533
    for e in protected:assert sha(Path(e['path']))==e['sha256']
    save(OUT/'summary.json',summary);save(OUT/'all_results.json',results);save(OUT/'rescued_gt_review.json',newlinks)
    case_results=[]
    for c in read(ROOT/'geometry/results/all_geometry_low_overlap_review_v1/cases.json'):
        ev=read(OUT/c['pair_id']/'eval_layout_experimental.json');un=read(OUT/c['pair_id']/'eval_hierarchy_union.json');gi=next(x['gt_index'] for x in ev['gt_coverage'] if x['original_gt_index']==c['original_gt_index']);case_results.append({'case_id':c['case_id'],'pair_id':c['pair_id'],'original_gt_index':c['original_gt_index'],'union_rescued':any(x['gt_index']==gi and x['covered'] for x in un['gt_coverage']),'experimental_rescued':any(x['gt_index']==gi and x['covered'] for x in ev['gt_coverage'])})
    save(OUT/'target_79_results.json',case_results)
    validation={'all_296_geometry_samples_scored':True,'gt_count_691_unchanged':True,'matching_and_original_groups_unchanged':True,'all_baseline_covered_gt_retained':True,'all_generation_completed_before_scoring':True,'no_gt_in_group_or_layout_generation':True,'protected_inputs_verified':len(protected),'human_review_complete':False}
    save(OUT/'validation.json',validation)
    report(summary,case_results,results)
    review_viewer()
    from geometry.scripts.review_hierarchical_results import main as review_results
    review_results()
    print(json.dumps({'summary':summary,'target79_union_rescued':sum(c['union_rescued'] for c in case_results),'target79_experimental_rescued':sum(c['experimental_rescued'] for c in case_results),'new_regions':sum(r['layout_proposals'] for r in results)},indent=2))

def report(summary,cases,results):
    lines=['# Hierarchical geometry — thử nghiệm toàn phạm vi 1005 mẫu','','296 mẫu có geometry / 691 GT chuẩn. Giữ nguyên detection, matching, classifier, GT và ngưỡng IoU≥0.30 hoặc CCR≥0.50. Cấu hình proposal được cố định trước scoring toàn tập. Dữ liệu đã được quan sát trong nghiên cứu; không có test held-out và không được tuyên bố tổng quát hóa.','','## Hai nhánh riêng','','1. Hierarchy union: nhóm theo mọi ancestor container và local gap thích ứng, vẫn yêu cầu chuyển động tương thích từng cặp; chấm union chính xác, không tính khoảng trống.','2. Layout extent — thử nghiệm: ≥3 thành viên không trùng, ≥2 OCR anchor exact, chuyển động nhất quán, spread≥0.5, vùng≤35% ảnh, thay đổi pixel >24 kênh trên ≥1% vùng và ≥90% vùng trong ảnh. Vùng container chỉ thêm khi layout con chiếm ≥50% diện tích parent ở cả hai phía. Đây là vùng suy ra gồm khoảng trống, KHÔNG phải actual union support. Tất cả vùng được dựng trước khi dùng GT.','','## Kết quả','','| Tập | Baseline coverage / PASS | Hierarchy union coverage / PASS | Layout thử nghiệm coverage / PASS |','|---|---|---|---|']
    for ds in ('first','second','all'):
        vals=[summary[ds][m] for m in ('baseline','hierarchy_union','layout_experimental')];lines.append('| '+ds+' | '+' | '.join(f"{v['covered']}/{v['gt_count']} ({100*v['coverage']:.2f}%) / {v['pass_samples']}/{v['samples']}" for v in vals)+' |')
    lines+=['','## 79 GT cần xử lý','','Hierarchy union cứu: '+str(sum(c['union_rescued'] for c in cases))+'/79. Layout thử nghiệm cứu: '+str(sum(c['experimental_rescued'] for c in cases))+'/79.','', 'Chẩn đoán trước thử nghiệm: union toàn bộ prediction thay đổi (bỏ qua tính hợp lệ nhóm, chỉ là upper bound) chỉ đạt ngưỡng cho 17/79; bbox bao toàn bộ prediction đạt 62/79 nhưng là phép gộp không hợp lệ, không dùng cứu điểm. Nhóm cũ có bbox đạt ngưỡng ở 20/79. Vì vậy không thể giải quyết tất cả chỉ bằng gom thêm thành viên union.','','Các vùng suy ra có thể bao phần tử không liên quan hoặc khoảng trống. Chưa có nhãn precision / review correctness; không chuyển tự động nhánh layout thành mặc định. Xem rescued_gt_review.json và index.html để kiểm tra bằng mắt. Giữ nguyên báo cáo baseline.','','Tái lập: `python3 -m geometry.scripts.experiment_hierarchical_geometry`.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

def review_viewer():
    cases=read(ROOT/'geometry/results/all_geometry_low_overlap_review_v1/cases.json')
    for c in cases:
        pid=c['pair_id'];g=read(OUT/pid/'groups_and_layout_proposals.json');gt=c['gt'];gb=[gt['x'],gt['y'],gt['x']+gt['w'],gt['y']+gt['h']]
        ev=read(OUT/pid/'eval_layout_experimental.json');c['improved_covered']=next(x['covered'] for x in ev['gt_coverage'] if x['original_gt_index']==c['original_gt_index'])
        for proposal in g['layout_proposals']:
            value=metrics([proposal['region']],gb)
            if value['intersection']<=0:continue
            c['candidates'].append({'kind':'inferred_layout','id':proposal['proposal_id'],'group_kind':proposal['evidence_type'],'prediction_indices':proposal['member_prediction_indices'],'before':[m['before_bbox'] for m in proposal['members']],'after':[m['after_bbox'] for m in proposal['members']],'regions':[proposal['region']],'before_ids':[m['before_id'] for m in proposal['members']],'after_ids':[m['after_id'] for m in proposal['members']],'pixel_evidence':proposal['pixel_evidence'],'actual_union':metrics(proposal['member_regions'],gb),**value})
        for group in g['groups']:
            value=metrics(group['member_regions'],gb)
            if value['intersection']<=0 or any(x['id']==group['group_id'] and x['kind']=='group' for x in c['candidates']):continue
            c['candidates'].append({'kind':'group','id':group['group_id'],'group_kind':group['kind'],'prediction_indices':group['member_prediction_indices'],'before':[m['before_bbox'] for m in group['members']],'after':[m['after_bbox'] for m in group['members']],'regions':group['member_regions'],'before_ids':[m['before_id'] for m in group['members']],'after_ids':[m['after_id'] for m in group['members']],**value})
        c['candidates'].sort(key=lambda p:(max(p['iou']/.3,p['ccr']/.5),p['iou']),reverse=True);c['candidate_count']=len(c['candidates'])
        for side in ('before','after'):c[side]=os.path.relpath(ROOT/'geometry/results/all_geometry_low_overlap_review_v1'/c[side],OUT)
    template=(ROOT/'geometry/scripts/all_low_overlap_review.html').read_text(encoding='utf-8').replace('63 mẫu · 79 GT thiếu overlap','Thử nghiệm hierarchy / layout · 79 GT').replace('Chỉ hiện GT bị bỏ sót có prediction/nhóm thay đổi giao GT nhưng IoU &lt; 0,30 và CCR &lt; 0,50.','79 GT missed của baseline; xem thêm vùng suy ra thử nghiệm, không phải xác nhận matching đúng.')
    template=template.replace("p.kind==='group'?'Nhóm':'Prediction'","p.kind==='inferred_layout'?'Vùng suy ra':p.kind==='group'?'Nhóm':'Prediction'")
    template=template.replace(' · FAIL`'," · ứng viên ${p.iou>=.3||p.ccr>=.5?'ĐẠT':'CHƯA ĐẠT'} · GT ${c.improved_covered?'được phủ mới (cần review)':'còn thiếu'}`")
    template=template.replace('Nhóm được vẽ từng envelope thành viên, không vẽ bbox bao nhóm thay cho union.','Nhóm: cam là từng envelope thành viên. Vùng suy ra: cam bao cả khoảng trống, không phải actual union support; xem actual_union trong bằng chứng.')
    template=template.replace("c.case_id+' · '+c.gt.selector","c.case_id+' · '+(c.improved_covered?'Phủ mới':'Còn thiếu')+' · '+c.gt.selector")
    template=template.replace('delta:p.delta,','delta:p.delta,pixel_evidence:p.pixel_evidence,actual_union:p.actual_union,')
    start=template.find('<p>Ảnh tổng quan before/after');end=template.find('</p>',start)
    if start>=0:template=template[:start]+template[end+4:]
    (OUT/'index.html').write_text(template.replace('/*CASES*/[]',json.dumps(cases,ensure_ascii=False).replace('<','\\u003c')), encoding='utf-8')
if __name__=='__main__':main()
