"""Post-score diagnostics, never used to construct or select proposals."""
import gzip
import json
from collections import Counter
from pathlib import Path
from geometry.pipeline import classify_change,prediction_envelope
from shared.regions import enclosure,area
from geometry.scripts.analyze_dataset2_failures import metrics,passes
from geometry.scripts.detect_dataset2 import ROOT,save
OUT=ROOT/'geometry/results/hierarchical_geometry_v1'

def main():
    cases=json.loads((ROOT/'geometry/results/all_geometry_low_overlap_review_v1/cases.json').read_text(encoding='utf-8'));target={r['case_id']:r for r in json.loads((OUT/'target_79_results.json').read_text(encoding='utf-8'))};remaining=[];counts=Counter();risks=[]
    for c in cases:
        pid=c['pair_id'];g=c['gt'];gt=[g['x'],g['y'],g['x']+g['w'],g['y']+g['h']];groups=json.loads((OUT/pid/'groups_and_layout_proposals.json').read_text(encoding='utf-8'))
        if target[c['case_id']]['experimental_rescued']:
            for p in groups['layout_proposals']:
                m=metrics([p['region']],gt)
                if passes(m) and m['prediction_area']/area(gt)>10:risks.append({'case_id':c['case_id'],'proposal_id':p['proposal_id'],'region_area_gt_ratio':m['prediction_area']/area(gt),'iou':m['iou'],'ccr':m['ccr'],'reason':'Region >10x GT area; CCR rescue requires semantic review.'})
            continue
        source=ROOT/'geometry/results' if c['dataset']=='first' else ROOT/'geometry/results/dataset2_from502_detection_v1'
        r=json.loads((source/pid/'matching_result_full.json').read_text(encoding='utf-8')) if c['dataset']=='first' else json.loads(gzip.open(source/pid/'matching_result_full.json.gz','rt').read());changed=[m for m in r['matched_pairs'] if classify_change(m['delta'],'edges8')]
        extent_groups=[p for p in groups['groups'] if passes(metrics([p['prediction_region']],gt))]
        if extent_groups:reason='group_extent_reaches_gt_but_independent_inference_guards_not_satisfied'
        elif changed and passes(metrics([enclosure([prediction_envelope(m['before_bbox'],m['after_bbox']) for m in changed])],gt)):reason='only_unrestricted_global_extent_reaches_gt_no_qualified_local_group'
        else:reason='even_global_changed_prediction_extent_does_not_reach_threshold'
        counts[reason]+=1;remaining.append({'case_id':c['case_id'],'reason':reason,'gt':g,'extent_passing_groups':len(extent_groups)})
    save(OUT/'remaining_69_analysis.json',{'counts':dict(counts),'cases':remaining});save(OUT/'large_region_review_flags.json',risks)
    result=json.loads((OUT/'all_results.json').read_text(encoding='utf-8'))
    for row in result:
        pid=row['pair_id'];baseline=ROOT/'geometry/results/group_coverage_v1' if row['dataset']=='first' else ROOT/'geometry/results/dataset2_same_gt_protocol_v1'
        b=json.loads((baseline/pid/'eval_edges8_union_support.json').read_text(encoding='utf-8'));u=json.loads((OUT/pid/'eval_hierarchy_union.json').read_text(encoding='utf-8'));x=json.loads((OUT/pid/'eval_layout_experimental.json').read_text(encoding='utf-8'))
        get=lambda e:{r['original_gt_index'] for r in e['gt_coverage'] if r['covered']}
        assert get(b)<=get(u)<=get(x)
    v=json.loads((OUT/'validation.json').read_text(encoding='utf-8'));v['baseline_gt_inclusion_checked_sample_by_sample']=True;save(OUT/'validation.json',v)
    marker='\n## 69 GT còn thiếu và rủi ro vùng lớn'
    p=OUT/'REPORT.md';p.write_text(p.read_text(encoding='utf-8').split(marker)[0]+marker+'\n\n```json\n'+json.dumps(dict(counts),ensure_ascii=False,indent=2)+'\n```\n\nCó '+str(len({r['case_id'] for r in risks}))+' GT phủ thêm có vùng suy ra >10 lần diện tích GT. Xem `large_region_review_flags.json`; đặc biệt 704/GT14 và 704/GT16. Đây là diagnostic sau chấm, không phải quy tắc sinh proposal. Chưa xác nhận đúng ngữ nghĩa.\n\nẢnh review: `rescued_review_1.jpg`, `rescued_review_2.jpg` nếu đã được tạo.\n', encoding='utf-8')
    save(OUT/'run_status.json',{'status':'complete_experimental','geometry_samples':296,'gt_count':691,'target79_rescued':10,'target79_remaining':69,'human_semantic_review_complete':False})
    print(json.dumps({'remaining':dict(counts),'risk_case_ids':sorted({r['case_id'] for r in risks})},indent=2))
if __name__=='__main__':main()
