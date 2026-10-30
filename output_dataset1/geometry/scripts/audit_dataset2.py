"""Read-only source audit before reusing the frozen geometry detector/matcher.

Does not reinterpret DOM as detector output or paired GT as legacy envelopes.
"""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from geometry.pipeline import DEFAULT_CONFIG
from geometry.prediction_groups import GROUP_CONFIG

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'geometry/results/dataset2_from502_v1'
DATA = ROOT / 'dataset2'
GT = ROOT / 'groundtruth/groundtruth'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = OUT / 'data_manifest.json'
    if previous.exists() and read(previous).get('dataset_root') != str(DATA):
        raise RuntimeError('Output belongs to a different dataset')
    protected = []
    for base in (DATA, GT, ROOT/'geometry/results/group_coverage_v1', ROOT/'geometry/results/group_coverage_organized_v1'):
        for p in sorted(base.rglob('*')):
            if p.is_file():
                protected.append({'path':str(p), 'sha256':digest(p)})
    for base in (ROOT/'baseline/json', ROOT/'inject/json', ROOT/'geometry/pairs'):
        for p in sorted(base.rglob('*.json')):
            protected.append({'path':str(p), 'sha256':digest(p)})
    byid = defaultdict(list)
    gt_index = defaultdict(list)
    unknown=[]
    for p in sorted(GT.glob('*.json')):
        if p.stem.isdigit(): gt_index[int(p.stem)].append(p)
    for p in sorted(DATA.iterdir()):
        if not p.is_dir():continue
        m=re.fullmatch(r'(\d+)_(.+)',p.name)
        if m:byid[int(m[1])].append(p)
        else:unknown.append(str(p))
    rows=[]
    for pid, folders in sorted(byid.items()):
        row={'pair_id':str(pid),'directories':[str(p) for p in folders],'issues':[],'warnings':[]}
        rows.append(row)
        if pid<502:
            row['status']='excluded_id_below_502';continue
        if len(folders)!=1:
            row['status']='invalid';row['issues'].append('duplicate_data_id');continue
        folder=folders[0]; gps=gt_index.get(pid,[])
        if len(gps)!=1:
            row['status']='invalid';row['issues'].append('missing_or_duplicate_gt');continue
        gp=gps[0];row['gt_path']=str(gp)
        try:
            g=read(gp);meta=read(folder/'metadata.json')
            if int(g['pair_id'])!=pid or int(meta['pair_id'])!=pid:row['issues'].append('metadata_or_gt_id_mismatch')
            boxes={s:[b for b in g.get(s+'_bboxes',[]) if 'geometry' in b.get('type',[])] for s in ('before','after')}
            row['geometry_gt_before']=len(boxes['before']);row['geometry_gt_after']=len(boxes['after'])
            row['gt_schema']=list(g)
            row['geometry_boxes']=boxes
            if not any(boxes.values()):row['status']='excluded_no_geometry';continue
            row['images']={};row['detection_candidates']={}
            for side,imgname in [('before','baseline.png'),('after','inject.png')]:
                ip=folder/imgname
                with Image.open(ip) as im:
                    im.verify()
                with Image.open(ip) as im:size=im.size
                row['images'][side]={'path':str(ip),'size':size}
                vp=meta.get('environment',{}).get('viewport',{})
                scale=meta.get('environment',{}).get('device_scale_factor',1)
                if vp and size!=(round(vp['width']*scale),round(vp['height']*scale)):
                    row['warnings'].append(side+'_image_differs_from_metadata_viewport')
                for b in boxes[side]:
                    vals=[b[k] for k in ('x','y','w','h')]
                    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in vals) or b['w']<=0 or b['h']<=0:
                        row['issues'].append(side+'_invalid_gt_bbox')
                    elif b['x']<0 or b['y']<0 or b['x']+b['w']>size[0] or b['y']+b['h']>size[1]:
                        row['warnings'].append(side+'_gt_outside_viewport_preserved')
                dp=folder/f'elements_{side}.json';d=read(dp)
                if isinstance(d,list):
                    schema='dom_elements' if all(isinstance(e,dict) and isinstance(e.get('bbox'),dict) for e in d) else 'unknown_list'
                elif isinstance(d,dict) and isinstance(d.get('elements'),list):schema='detector_candidate'
                else:schema='unknown'
                row['detection_candidates'][side]={'path':str(dp),'schema':schema,'count':len(d) if isinstance(d,list) else len(d.get('elements',[]))}
                if schema!='detector_candidate':row['issues'].append(side+'_missing_frozen_detector_output')
                upstream=ROOT/('baseline' if side=='before' else 'inject')/'json'
                row['detection_candidates'][side]['existing_fused_files']=[str(p) for p in upstream.glob(f'{pid:03d}_*_full_hierarchy.json')]
            if 'bboxes' not in g:row['issues'].append('paired_gt_requires_verified_legacy_envelope_conversion')
            row['status']='blocked_missing_compatible_inputs' if row['issues'] else 'ready'
            row['warnings']=sorted(set(row['warnings']))
        except Exception as exc:
            row['status']='invalid';row['issues'].append(f'{type(exc).__name__}: {exc}')
    cache_cfg=read(ROOT/'geometry/results/000/matching_result_full.json')['config']
    differences={k:{'current':v,'frozen':cache_cfg.get(k)} for k,v in DEFAULT_CONFIG.items() if cache_cfg.get(k)!=v}
    frozen_groups=read(ROOT/'geometry/results/group_coverage_v1/group_config.json')
    config={'matching':cache_cfg,'grouping':frozen_groups,'primary_classifier':'edges8','primary_mode':'union_support','current_matching_defaults_differ':differences,'current_group_defaults_equal':GROUP_CONFIG==frozen_groups,'migration_regression':read(ROOT/'docs/migrations/2026-10-21/regression_validation.json')}
    statuses=Counter(r['status'] for r in rows)
    selected=[r for r in rows if r.get('geometry_gt_before',0) or r.get('geometry_gt_after',0)]
    summary={'total_dataset_directories':sum(map(len,byid.values())),'unrecognized_directories':unknown,'min_id':min(byid),'max_id':max(byid),'status_counts':dict(statuses),'geometry_samples':len(selected),'geometry_before_boxes':sum(r['geometry_gt_before'] for r in selected),'geometry_after_boxes':sum(r['geometry_gt_after'] for r in selected),'evaluated_samples':0,'coverage':None,'gt_without_dataset_id_ge502':[str(p) for i,ps in gt_index.items() if i>=502 and i not in byid for p in ps],'issue_counts':dict(Counter(x for r in rows for x in r['issues']))}
    objects={'data_manifest.json':{'dataset_root':str(DATA),'gt_root':str(GT),'min_id_inclusive':502,'samples':rows},'inventory.json':summary,'frozen_config.json':config,'protected_inputs.json':protected,'run_status.json':{'status':'blocked_missing_compatible_inputs','completed_pairs':[],'selected_pairs':[r['pair_id'] for r in selected],'reason':'DOM is not frozen OCR/YOLO/CV output; paired GT is not legacy envelope schema.'}}
    for name,obj in objects.items():(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    unchanged=all(digest(Path(e['path']))==e['sha256'] for e in protected)
    assert unchanged
    (OUT/'validation.json').write_text(json.dumps({'protected_files':len(protected),'source_and_old_results_preserved':unchanged,'matching_defaults_equal_frozen':not differences,'group_defaults_equal_frozen':GROUP_CONFIG==frozen_groups,'matching_or_scoring_executed':False},indent=2)+'\n', encoding='utf-8')
    lines=['# Dataset2 — kiểm kê trước đánh giá geometry','',f"Phạm vi: ID ≥ 502. Nguồn: `{DATA}`. GT: `{GT}`.",'',f"Có {summary['total_dataset_directories']} thư mục, ID {summary['min_id']}–{summary['max_id']}. Trạng thái: {dict(statuses)}.",f"Có {len(selected)} mẫu geometry, {summary['geometry_before_boxes']} bbox before và {summary['geometry_after_boxes']} bbox after. Chưa quy đổi hai danh sách này thành số vùng GT envelope.",'','## Chưa thể chạy đúng pipeline chuẩn','','1. Dataset2 chứa ảnh và DOM (`elements_before/after.json`), không có fused detection OCR/YOLO/CV tương ứng. Cache baseline/inject hiện có chỉ đến ID 500. Không chuyển DOM thành detection vì sẽ thay đầu vào và đặc tính pipeline.','2. GT mới có `before_bboxes` và `after_bboxes`; evaluator chuẩn chỉ đọc `bboxes`. Đưa trực tiếp GT này vào evaluator sẽ tạo 0 GT và có thể báo PASS sai. Cần xác minh quy tắc ghép các endpoint để tạo envelope tương đương GT cũ, gồm trường hợp selector trùng hoặc số lượng hai phía khác nhau.','', 'Không chạy matching/scoring; không có chỉ số coverage, PASS/FAIL hoặc so sánh chất lượng hợp lệ. Không coi thiếu input là detection fail.','', '## Cấu hình và provenance','','Đã lưu cấu hình từ cache chuẩn, cấu hình nhóm lịch sử, regression sau tách project, hash input và kết quả cũ. Xem frozen_config.json và validation.json.','', 'Upstream perception được tìm thấy tại `/home/tung/Desktop/RPA/OCR + Yolo26`; có sử dụng Google Cloud Vision bên ngoài và model YOLO. Chưa xác minh phiên bản/cấu hình upstream tương đương lần tạo tập cũ nên chưa khởi chạy hàng loạt.','', '## Tái kiểm tra','','`python3 -m geometry.scripts.audit_dataset2`','','## Mẫu geometry trong phạm vi','','| ID | GT before | GT after | Trạng thái |','|---|---:|---:|---|']
    lines += [f"| {r['pair_id']} | {r['geometry_gt_before']} | {r['geometry_gt_after']} | {r['status']} |" for r in selected]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
