"""Reuse frozen geometry predictions with the exact canonical dataset1 GT protocol."""
import ast
import csv
import gzip
import json
import math
import os
from pathlib import Path
from PIL import Image
from geometry import pipeline
from geometry.prediction_groups import GROUP_CONFIG,build_prediction_groups
from geometry.scripts.detect_dataset2 import ROOT,save,sha
GT=ROOT.parent/'dataset1/thesis-dataset-main/groundtruth'
SOURCE=ROOT/'geometry/results/dataset2_from502_detection_v1'
BASE=ROOT/'geometry/results/group_coverage_v1'
OUT=ROOT/'geometry/results/dataset2_same_gt_protocol_v1'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def geometry_boxes(d):
    if not isinstance(d,dict) or not isinstance(d.get('bboxes'),list):raise ValueError('Canonical GT requires bboxes list')
    selected=[]
    for i,b in enumerate(d['bboxes']):
        if 'geometry' not in b.get('type',[]):continue
        v=[b.get(k) for k in ('x','y','w','h')]
        if not all(isinstance(n,(int,float)) and math.isfinite(n) for n in v) or b['w']<=0 or b['h']<=0:raise ValueError(f'Invalid GT bbox {i}')
        selected.append({'original_index':i,'annotation':b})
    return selected

def semantic_function(path,name):
    module=ast.parse(Path(path).read_text(encoding='utf-8'))
    function=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name==name)
    for node in ast.walk(function):
        if isinstance(node,ast.ImportFrom) and node.module=='prediction_groups':node.module='geometry.prediction_groups'
    return ast.dump(function,include_attributes=False)

def validate_ev(ev,boxes):
    assert ev['gt_count']==len(boxes)>0
    assert ev['covered']+len(ev['missed_gt'])==len(boxes)
    assert ev['pass']==(ev['covered']==len(boxes))
    assert [r['original_gt_index'] for r in ev['gt_coverage']]==[b['original_index'] for b in boxes]

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'data_manifest.json').exists() and read(OUT/'data_manifest.json')['gt_root']!=str(GT):raise ValueError('Existing output belongs to another source')
    base_cfg=read(ROOT/'geometry/results/000/matching_result_full.json')['config'];match_cfg={k:v for k,v in base_cfg.items() if k!='image_root'}
    group_cfg=read(BASE/'group_config.json');assert group_cfg==GROUP_CONFIG
    old_src=ROOT/'docs/migrations/2026-10-21/retired_new_pipeline.py'
    legacy_equal={f:semantic_function(old_src,f)==semantic_function(ROOT/'geometry/pipeline.py',f) for f in ('classify_change','prediction_envelope','evaluate','evaluate_grouped')}
    assert all(legacy_equal.values())
    oldids=read(ROOT/'geometry/results/matching_manifest.json')['pair_ids']
    assert len(oldids)==131
    for pid in oldids:assert read(GT/f'{pid}.json')==read(ROOT/'geometry/pairs'/pid/'groundtruth.json')
    protected=[]
    for p in sorted(GT.glob('*.json')):protected.append({'path':str(p),'sha256':sha(p)})
    for folder in (SOURCE,ROOT/'geometry/results/dataset2_native_gt_v1',ROOT/'geometry/results/dataset2_gt_review',BASE):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and not p.is_symlink():protected.append({'path':str(p),'sha256':sha(p)})
    folders={}
    for p in (ROOT/'dataset2').iterdir():
        if p.is_dir():
            pid=int(p.name.split('_')[0]);assert pid not in folders;folders[pid]=p
    survey=[];differences=[];selected=[];canonical_ids=[];paired_ids=[]
    for pid in range(502,1006):
        folder=folders[pid];gp=GT/f'{pid:03d}.json';gt=read(gp);assert int(gt['pair_id'])==pid
        boxes=geometry_boxes(gt);paired_path=ROOT/'groundtruth/groundtruth'/f'{pid:03d}.json';paired=read(paired_path)
        paired_count={s:sum('geometry' in b.get('type',[]) for b in paired[s+'_bboxes']) for s in ('before','after')}
        if boxes:canonical_ids.append(str(pid))
        if sum(paired_count.values()):paired_ids.append(str(pid))
        comparison={'pair_id':str(pid),'canonical_geometry':len(boxes),'paired_geometry_before':paired_count['before'],'paired_geometry_after':paired_count['after']}
        if bool(boxes)!=bool(sum(paired_count.values())):comparison.update(canonical=gt,paired=paired)
        differences.append(comparison)
        row={'pair_id':str(pid),'gt_path':str(gp),'gt_sha256':sha(gp),'total_bboxes':len(gt['bboxes']),'geometry_count':len(boxes),'geometry_annotations':boxes,'folder':str(folder),'status':'selected' if boxes else 'excluded_no_geometry'}
        if boxes:
            row['matching_path']=str(SOURCE/str(pid)/'matching_result_full.json.gz');row['groups_path']=str(SOURCE/str(pid)/'groups_edges8.json')
            row['images']={s:str(folder/n) for s,n in [('before','baseline.png'),('after','inject.png')]}
            row['detections']={s:str(ROOT/'dataset2_detection'/s/'json'/f'{folder.name}_full_hierarchy.json') for s in ('baseline','inject')}
            selected.append(row)
        survey.append(row)
    assert len(selected)==165 and sum(r['geometry_count'] for r in selected)==421
    save(OUT/'data_manifest.json',{'gt_root':str(GT),'scope':[502,1005],'source_schema':'bboxes','samples':survey,'selected_samples':165,'geometry_gt_count':421,'excluded_no_geometry':len(survey)-len(selected),'outside_id_scope':[str(i) for i in folders if i<502 or i>1005]})
    diff={'canonical_gt_root':str(GT),'paired_gt_root':str(ROOT/'groundtruth/groundtruth'),'canonical_geometry_ids':canonical_ids,'paired_geometry_ids':paired_ids,'canonical_only':sorted(set(canonical_ids)-set(paired_ids),key=int),'paired_only':sorted(set(paired_ids)-set(canonical_ids),key=int),'per_sample':differences}
    save(OUT/'gt_source_comparison.json',diff)
    save(OUT/'frozen_config.json',{'matching':base_cfg,'grouping':group_cfg,'classifier':'edges8','primary_mode':'union_support','thresholds':{'iou':.3,'ccr':.5},'legacy_evaluator_functions_ast_equal':legacy_equal})
    prepared=[]
    # Verify every cached prediction and independently reproduce groups before scoring.
    for row in selected:
        pid=row['pair_id'];prov=read(SOURCE/pid/'prediction_provenance.json');assert prov['config']==match_cfg
        assert prov['pipeline_sha256']==sha(ROOT/'geometry/pipeline.py') and prov['shared_sha256']==sha(ROOT/'shared/matching.py') and prov['group_sha256']==sha(ROOT/'geometry/prediction_groups.py')
        for p,h in prov['inputs'].items():assert sha(Path(p))==h
        for side in ('baseline','inject'):
            dp=Path(row['detections'][side]);receipt=read(ROOT/'dataset2_detection'/side/'receipts'/f"{Path(row['folder']).name}.json")
            assert receipt['output_sha256']==sha(dp)
            image=Path(row['images']['before' if side=='baseline' else 'after']);assert sha(image)==receipt['image_sha256']
        with gzip.open(row['matching_path'],'rt') as f:r=json.load(f)
        assert {k:v for k,v in r['config'].items() if k!='image_root'}==match_cfg
        for side in ('before','after'):assert Path(r['image_paths'][side]).resolve()==Path(row['images'][side]).resolve()
        groups=read(row['groups_path']);assert groups['config']==group_cfg and groups['uses_groundtruth'] is False
        with Image.open(row['images']['before']) as im:size=im.size
        predictions=[m for m in r['matched_pairs'] if pipeline.classify_change(m['delta'],'edges8')]
        assert groups==build_prediction_groups(predictions,r['matched_pairs'],size,group_cfg)
        prepared.append((row,r,groups))
    save(OUT/'prediction_validation.json',{'samples_verified':len(prepared),'all_groups_reproduced_identically_without_gt':True,'matching_config_and_code_hashes_verified':True,'all_detector_receipts_verified':True,'gt_not_used_for_prediction_generation':True})
    def score(item):
        row,r,groups=item;pid=row['pair_id'];gp=Path(row['gt_path']);boxes=row['geometry_annotations'];out=OUT/pid
        single=pipeline.evaluate(r['matched_pairs'],gp,'edges8');ev=pipeline.evaluate_grouped(r['matched_pairs'],gp,'edges8',groups,'union_support')
        for e in (single,ev):validate_ev(e,boxes)
        assert ev['covered']>=single['covered']
        save(out/'eval_edges8_single_envelope.json',single);save(out/'eval_edges8_union_support.json',ev)
        return {'pair_id':pid,'gt_count':ev['gt_count'],'covered':ev['covered'],'missed':len(ev['missed_gt']),'pass':ev['pass'],'single_covered':single['covered'],'predictions':ev['predictions'],'groups':ev['group_count'],'group_rescued':len(ev['group_rescued_gt']),'unverified':ev['unverified'],'missed_gt':ev['missed_gt']}
    smoke=[score(x) for x in prepared[:5]];save(OUT/'smoke5_results.json',smoke)
    results=smoke+[score(x) for x in prepared[5:]]
    summary={'scope':[502,1005],'samples_in_scope':404,'excluded_no_geometry':239,'samples_requested':165,'samples_scored':len(results),'errors':[],
             'gt_count':sum(r['gt_count'] for r in results),'covered':sum(r['covered'] for r in results),'missed':sum(r['missed'] for r in results),'pass_samples':sum(r['pass'] for r in results),'fail_samples':sum(not r['pass'] for r in results),
             'predictions':sum(r['predictions'] for r in results),'groups':sum(r['groups'] for r in results),'single_covered':sum(r['single_covered'] for r in results),'group_rescued':sum(r['group_rescued'] for r in results)}
    summary['coverage']=summary['covered']/summary['gt_count'];assert summary['gt_count']==421 and summary['samples_scored']==165
    for name,obj in [('summary.json',summary),('all_results.json',results),('failure_samples.json',[r for r in results if not r['pass']])]:save(OUT/name,obj)
    with (OUT/'evaluation.csv').open('w') as f:
        keys=[k for k in results[0] if k!='missed_gt'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in results)
    for e in protected:assert sha(Path(e['path']))==e['sha256']
    save(OUT/'run_manifest.json',{'gt_root':str(GT),'protected_inputs':protected,'pair_ids':canonical_ids,'config_sha256':sha(OUT/'frozen_config.json'),'code_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'geometry/pipeline.py',ROOT/'shared/matching.py',ROOT/'geometry/prediction_groups.py',Path(__file__)]},'detector_rerun':False,'matching_rerun':False,'groups_reproduced_without_gt':True,'historical_detector_version_hash_available':False})
    save(OUT/'validation.json',{'canonical_gt_165_samples_421_boxes':True,'all_131_baseline_gt_files_identical_to_canonical':True,'legacy_evaluator_functions_ast_equal':legacy_equal,'all_groups_reproduced_identically':True,'protected_files_unchanged':len(protected),'all_gt_scored_without_transformation':True,'prediction_annotations_count_and_original_indices_verified':True,'human_semantic_review_complete':False})
    report(summary,results,diff)
    viewer(prepared,results)
    save(OUT/'run_status.json',{'status':'complete','samples_scored':len(results),'errors':[]})
    print(json.dumps(summary,indent=2))

def report(s,results,diff):
    lines=['# Dataset2 — cùng nguồn GT và evaluator với tập đầu','','Nguồn GT duy nhất: `'+str(GT)+'`. Đọc nguyên `bboxes`, chỉ lọc geometry; không chuyển GT DOM hai phía, không gộp/tách, không sửa bbox, vẫn giữ đa nhãn/fallback/ngoài viewport.','','## Kết quả so sánh — edges8 + union_support','','| Chỉ số | Tập đầu | Tập sau |','|---|---:|---:|','| Phạm vi ID | 000–500 | 502–1005 |',f"| Mẫu geometry | 131 | {s['samples_scored']} |",f"| GT geometry | 270 | {s['gt_count']} |",f"| GT covered / missed | 220 / 50 | {s['covered']} / {s['missed']} |",f"| Coverage | 81,48% | {100*s['coverage']:.2f}% |",f"| PASS / FAIL | 98 / 33 | {s['pass_samples']} / {s['fail_samples']} |",f"| Prediction / nhóm | 1661 / 550 | {s['predictions']} / {s['groups']} |",f"| Envelope đơn: GT covered | 210 | {s['single_covered']} |",f"| Nhóm phủ thêm GT | 10 | {s['group_rescued']} |",'','GT/đơn vị và evaluator hiện thống nhất. So sánh mô tả hai tập khác nhau; không coi chênh lệch là thay đổi chất lượng thuật toán. Coverage không phải precision correspondence. Detector tập sau tái dùng entrypoint tập đầu hiện có, nhưng không có hash lịch sử upstream để khẳng định detector bit-for-bit tương đương. Matching/grouping config, code hashes và groups tái lập đều đã kiểm chứng.','','## Khác biệt hai nguồn GT','','Nguồn chuẩn có 165 mẫu geometry; nguồn hai phía có 166. ID chỉ có geometry ở nguồn hai phía: '+', '.join(diff['paired_only'])+'. Nguồn chuẩn không có geometry tại 522, chỉ có một bbox style. Không lấy nhãn geometry từ nguồn khác để bổ sung. Xem `gt_source_comparison.json` để đối chiếu đủ tọa độ, selector và nhãn.','','Phạm vi gồm 404 ID; 239 mẫu không có geometry bị loại đúng phạm vi. ID 501 ngoài phạm vi. Không có mẫu lỗi, bị bỏ âm thầm, hoặc PASS vì GT rỗng. Các báo cáo 509/799 và 212/298 là protocol/nguồn GT khác, không thay thế kết quả này.','','## Tái lập','','`python3 -m geometry.scripts.evaluate_dataset2_same_gt`','','Không chạy lại detector hoặc matcher. Tái xác minh input, reproduce groups độc lập GT, kiểm tra 5 mẫu trước khi chấm toàn bộ.','','## Từng mẫu','','| ID | GT | Covered | Missed | PASS |','|---|---:|---:|---:|---|']
    lines += [f"| {r['pair_id']} | {r['gt_count']} | {r['covered']} | {r['missed']} | {r['pass']} |" for r in results]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (OUT/'GT_SOURCE_COMPARISON.md').write_text('# Đối chiếu nguồn GT\n\nGT chuẩn có `bboxes`; nguồn khác có `before_bboxes`/`after_bboxes`. Không trộn hai nguồn.\n\n```json\n'+json.dumps([r for r in diff['per_sample'] if 'canonical' in r],ensure_ascii=False,indent=2)+'\n```\n', encoding='utf-8')

def viewer(prepared,results):
    payload=[]
    for (row,r,g),res in zip(prepared,results):
        ev=read(OUT/row['pair_id']/'eval_edges8_union_support.json');annotations={x['original_index']:x['annotation'] for x in row['geometry_annotations']}
        payload.append({'id':row['pair_id'],'summary':res,'before':os.path.relpath(row['images']['before'],OUT),'after':os.path.relpath(row['images']['after'],OUT),
                        'gt':[dict(annotations[c['original_gt_index']],original_index=c['original_gt_index'],covered=c['covered'],prediction_indices=c['prediction_indices'],group_ids=c.get('group_ids',[])) for c in ev['gt_coverage']],
                        'predictions':[dict(before=m['before_bbox'],after=m['after_bbox'],envelope=pipeline.prediction_envelope(m['before_bbox'],m['after_bbox'])) for m in r['matched_pairs'] if pipeline.classify_change(m['delta'],'edges8')]})
    template=(ROOT/'geometry/scripts/dataset2_same_gt_review.html').read_text(encoding='utf-8')
    (OUT/'index.html').write_text(template.replace('/*PAYLOAD*/[]',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')), encoding='utf-8')

if __name__=='__main__':main()
