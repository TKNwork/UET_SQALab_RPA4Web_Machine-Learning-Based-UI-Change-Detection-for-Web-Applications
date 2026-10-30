"""Publish the current group-coverage run and its matching cache to the viewer."""
import hashlib
import json
from shared.paths import resolve_recorded_path
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'geometry/results'
OUT=SOURCE/'group_coverage_organized_v1'

def main(publish=False):
    manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    if json.loads((OUT/'run_status.json').read_text(encoding='utf-8'))['status']!='complete':raise RuntimeError('Incomplete run')
    summary,ids=manifest['summary'],manifest['pair_ids']
    rescues=json.loads((OUT/'rescued_gt_review.json').read_text(encoding='utf-8'))
    failures=json.loads((OUT/'failure_samples.json').read_text(encoding='utf-8'))
    lines=['# Prediction groups — kết quả hiện hành','',f"Đã đánh giá {len(ids)} mẫu; tạo toàn bộ nhóm trước khi đọc GT. Giữ nguyên matched pairs và classifier. Nhóm mới chưa được xác nhận đúng bằng review thủ công.",'',
           '## Quy tắc','',
           '- Shared parent: con thuộc cùng container tương ứng ở hai ảnh; chọn parent nhỏ nhất.',
           '- Local motion: các thành viên lân cận hai phía, chuyển động tương thích từng cặp.',
           '- Loại bằng chứng duplicate; ít nhất 2 thành viên. Nhóm không bao quá 35% diện tích ảnh.',
           '- union_support: diện tích hợp chính xác của các envelope thành viên; IoU≥0.30 hoặc CCR≥0.50. Đây là mặc định CLI.',
           '- structural_proxy: thử nghiệm dùng thêm bbox phạm vi nhóm; ít nhất 3 thành viên, spread≥0.50, ít nhất 10% GT có union support và bbox vượt ngưỡng. Cần review vì có thể bao khoảng trống.',
           '- Một GT được đáp ứng bởi prediction đơn hoặc nhóm. Không assignment prediction–GT one-to-one; uncertain không được chấm.',
           '', '## Kết quả edges8','', '| Mode | GT covered | Coverage | PASS | FAIL | Nhóm |','|---|---:|---:|---:|---:|---:|']
    for mode,s in summary['edges8'].items():lines.append(f"| {mode} | {s['covered']}/{s['gt_count']} | {100*s['covered']/s['gt_count']:.1f}% | {s['pass_samples']}/{s['samples']} | {s['samples']-s['pass_samples']} | {s['groups']} |")
    lines+=['','Envelope đơn là thành phần/chế độ đối chiếu hiện hành của thuật toán nhóm, không phải pipeline matcher cũ. Unverified không có nghĩa false positive. Coverage không phải precision của correspondence hoặc nhóm.', '', '## GT được nhóm phủ thêm', '', '| Mode | ID/GT | Review thủ công |', '|---|---|---|']
    for r in rescues:lines.append(f"| {r['mode']} | {r['pair_id']}/GT{r['gt_index']} | Chưa có |")
    lines+=['','## Mẫu FAIL — union_support','','| ID | Covered/GT | GT thiếu |','|---|---:|---|']
    for r in failures['union_support']:lines.append(f"| {r['pair_id']} | {r['covered']}/{r['gt_count']} | "+', '.join(map(str,r['missed_gt']))+' |')
    lines+=['','## Tái lập','','```bash','python3 -m geometry.scripts.evaluate_group_coverage','python3 -m geometry.scripts.build_group_report','python3 -m unittest discover -s tests -v','```','',
            'Matched cache: geometry/results/<ID>/matching_result_full.json, metadata: matching_manifest.json. Kết quả chạy hiện hành: group_coverage_organized_v1. Chi tiết file/schema và sơ đồ: docs/PROJECT_STRUCTURE.md.','',
            '## Giới hạn','',
            'Parent từ bbox không phải cây DOM xác nhận. Nhóm chuyển động có thể trộn các sự kiện khác nhau. Threshold grouping chưa hiệu chuẩn độc lập. Envelope thành viên có thể chứa khoảng trống chuyển động. Chưa có nhãn correctness nên không báo precision/F1.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Validate provenance before optional publication. Historical public data is
    # untouched by a report-only run after the folder migration.
    for entry in manifest['inputs']:
        assert hashlib.sha256(resolve_recorded_path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
    for file,key in [('geometry/prediction_groups.py','group_code_sha256'),('geometry/pipeline.py','evaluator_code_sha256')]:
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==manifest[key]
    if not publish:
        (OUT/'validation.json').write_text(json.dumps({'samples':len(ids),'source_hashes_preserved':True,'grouping_completed_before_gt_scoring':True,'human_group_review_complete':False,'published':False},indent=2), encoding='utf-8')
        print('Report written:',OUT/'REPORT.md')
        return
    public=ROOT/'geometry/report_viewer/public/data/geometry-v3';public.mkdir(parents=True,exist_ok=True)
    cases=[]
    for pid in ids:
        dest=public/pid;dest.mkdir(exist_ok=True)
        r=json.loads((SOURCE/pid/'matching_result_full.json').read_text(encoding='utf-8'))
        r['evidence_detail']['cost_matrix_info'].pop('candidate_trace',None)
        r['candidate_trace_omitted_for_viewer']=True
        (dest/'matching_result_full.json').write_text(json.dumps(r,ensure_ascii=False), encoding='utf-8')
        for side,path in r['image_paths'].items():
            target=dest/(side+'.png');original=Path(path)
            if not target.exists() or target.stat().st_size!=original.stat().st_size:shutil.copy2(original,target)
        for mode in ('single_envelope','union_support','structural_proxy'):
            ev_path=OUT/pid/f'eval_edges8_{mode}.json';ev=json.loads(ev_path.read_text(encoding='utf-8'))
            assert ev['covered']==sum(g['covered'] for g in ev['gt_coverage'])==len(ev['assignments'])
            shutil.copy2(ev_path,dest/f'eval_{mode}_edges8.json')
    for record in rescues:
        pid=record['pair_id']
        gt=[g for g in json.loads((ROOT/'geometry/pairs'/pid/'groundtruth.json').read_text(encoding='utf-8'))['bboxes'] if 'geometry' in g['type']][record['gt_index']]
        groups=json.loads((OUT/pid/'groups_edges8.json').read_text(encoding='utf-8'))
        cases.append(dict(record,id=f"{pid}-GT{record['gt_index']}",gt=gt,image_size=groups['image_size']))
    payload=json.dumps(cases,ensure_ascii=False,separators=(',',':')).replace('<',r'\u003c')
    target=ROOT/'geometry/report_viewer/public/group-review';target.mkdir(exist_ok=True)
    (target/'index.html').write_text((ROOT/'geometry/scripts/group_review_page.html').read_text(encoding='utf-8').replace('__GROUP_CASES__',payload), encoding='utf-8')
    shutil.copy2(OUT/'REPORT.md',public/'group_REPORT.md')
    (public/'manifest.json').write_text(json.dumps({'pair_ids':ids,'summary':summary['edges8'],'default_mode':'union_support'},indent=2), encoding='utf-8')
    for entry in manifest['inputs']:assert hashlib.sha256(resolve_recorded_path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
    for file,key in [('geometry/prediction_groups.py','group_code_sha256'),('geometry/pipeline.py','evaluator_code_sha256')]:assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==manifest[key]
    (OUT/'validation.json').write_text(json.dumps({'samples':len(ids),'source_hashes_preserved':True,'gt_coverage_assignments_consistent':True,'grouping_completed_before_gt_scoring':True,'human_group_review_complete':False},indent=2), encoding='utf-8')
    print(json.dumps(summary['edges8'],indent=2));print('Group review cases:',len(cases))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish',action='store_true',help='Explicitly refresh geometry viewer copies')
    main(publish=parser.parse_args().publish)
