"""Audit missed canonical GT using frozen evidence; conditions are not human root-cause labels."""
import csv
import gzip
import json
import os
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
from geometry import pipeline
from shared.regions import area,intersect,union_area
from geometry.scripts.detect_dataset2 import ROOT,save,sha
RUN=ROOT/'geometry/results/dataset2_same_gt_protocol_v1'
SOURCE=ROOT/'geometry/results/dataset2_from502_detection_v1'
OUT=RUN/'failure_analysis'
LABELS={
 'gt_outside_image':'GT nằm hoàn toàn ngoài ảnh',
 'changed_overlap_below_threshold':'Có prediction thay đổi/nhóm giao GT nhưng chưa đạt IoU/CCR',
 'uncertain_valid_change':'Có cặp uncertain thay đổi và envelope đạt ngưỡng GT',
 'matched_below_change_threshold':'Có matched chưa vượt ngưỡng geometry nhưng envelope đạt GT',
 'detection_without_correspondence':'Detection có đủ coverage ở hai phía nhưng thiếu correspondence thay đổi đáp ứng',
 'one_sided_detection':'Detection chỉ đủ coverage GT ở một phía before hoặc after',
 'detection_insufficient':'Detection không có đủ coverage vùng GT'}

def box(g):return [g['x'],g['y'],g['x']+g['w'],g['y']+g['h']]
def metrics(boxes,g):
    inter=union_area([intersect(b,g) for b in boxes]);support=union_area(boxes)
    return dict(iou=inter/max(area(g)+support-inter,1e-9),ccr=inter/max(area(g),1e-9),intersection=inter,prediction_area=support)
def passes(m):return m['iou']+1e-12>=.3 or m['ccr']+1e-12>=.5
def best(items):return max(items,key=lambda m:(max(m['iou']/.3,m['ccr']/.5),m['iou']),default=None)
def pairitem(m,g,index=None):
    return dict(kind='pair',index=index,before_id=m['before_id'],after_id=m['after_id'],before_bbox=m['before_bbox'],after_bbox=m['after_bbox'],delta=m.get('delta'),match_type=m.get('match_type'),quality_flags=m.get('quality_flags',[]),before_class=m.get('before_class'),after_class=m.get('after_class'),evidence=m.get('evidence',{}),uncertain_reason=m.get('uncertain_reason'),**metrics([pipeline.prediction_envelope(m['before_bbox'],m['after_bbox'])],g))

def main():
    OUT.mkdir(exist_ok=True);manifest=json.loads((RUN/'data_manifest.json').read_text(encoding='utf-8'));lookup={r['pair_id']:r for r in manifest['samples']}
    failures=json.loads((RUN/'failure_samples.json').read_text(encoding='utf-8'));protected=[];targets=[];samples=[];viewer=[];flags=Counter()
    for path in [RUN/'summary.json',RUN/'data_manifest.json',RUN/'all_results.json',RUN/'failure_samples.json']:
        protected.append({'path':str(path),'sha256':sha(path)})
    for fail in failures:
        pid=fail['pair_id'];row=lookup[pid];gp=Path(row['gt_path']);protected.append({'path':str(gp),'sha256':sha(gp)})
        with gzip.open(SOURCE/pid/'matching_result_full.json.gz','rt') as f:r=json.load(f)
        for name in ('matching_result_full.json.gz','groups_edges8.json'):
            p=SOURCE/pid/name;protected.append({'path':str(p),'sha256':sha(p)})
        ev=json.loads((RUN/pid/'eval_edges8_union_support.json').read_text(encoding='utf-8'));groups=json.loads((SOURCE/pid/'groups_edges8.json').read_text(encoding='utf-8'))
        changed=[m for m in r['matched_pairs'] if pipeline.classify_change(m['delta'],'edges8')]
        unchanged=[m for m in r['matched_pairs'] if not pipeline.classify_change(m['delta'],'edges8')]
        uncertain=[m for m in r['uncertain_pairs'] if pipeline.classify_change(m['delta'],'edges8')]
        raw={s:json.loads(Path(row['detections']['baseline' if s=='before' else 'inject']).read_text(encoding='utf-8'))['elements'] for s in ('before','after')}
        annotations={x['original_index']:x['annotation'] for x in row['geometry_annotations']};sample_targets=[]
        for target in ev['gt_coverage']:
            if target['covered']:continue
            gt=annotations[target['original_gt_index']];gb=box(gt)
            candidates=[pairitem(m,gb,i) for i,m in enumerate(changed)]
            group_candidates=[dict(kind='group',group_id=g['group_id'],member_prediction_indices=g['member_prediction_indices'],member_regions=g['member_regions'],**metrics(g['member_regions'],gb)) for g in groups['groups']]
            cb=best(candidates);gg=best(group_candidates);assert not any(passes(m) for m in candidates+group_candidates)
            ub=best([pairitem(m,gb) for m in uncertain]);mb=best([pairitem(m,gb) for m in unchanged])
            visibility={}
            for s in ('before','after'):
                with Image.open(row['images'][s]) as im:w,h=im.size
                visibility[s]={'image_size':[w,h],'gt_visible_fraction':area(intersect(gb,[0,0,w,h]))/area(gb)}
            raw_stats={}
            for s,elems in raw.items():
                single=[dict(id=e['id'],source=e.get('source'),bbox=e['bbox'],**metrics([e['bbox']],gb)) for e in elems]
                raw_stats[s]={'best_single':best(single),'union_all_detections':metrics([e['bbox'] for e in elems],gb),'count_intersecting':sum(x['intersection']>0 for x in single)}
            flags_here=[]
            if all(x['gt_visible_fraction']==0 for x in visibility.values()):flags_here.append('entirely_outside_viewport')
            elif any(x['gt_visible_fraction']<1 for x in visibility.values()):flags_here.append('partially_outside_viewport')
            if ub and passes(ub):flags_here.append('uncertain_changed_envelope_would_pass')
            if mb and passes(mb):flags_here.append('unchanged_matched_envelope_would_pass_if_marked_changed')
            if max((m['intersection'] for m in candidates+group_candidates),default=0)>0:flags_here.append('changed_prediction_or_group_intersects_gt')
            raw_has_single=any(v['best_single'] and passes(v['best_single']) for v in raw_stats.values())
            raw_has_union=any(passes(v['union_all_detections']) for v in raw_stats.values())
            if raw_has_union and not raw_has_single:flags_here.append('only_union_of_raw_detections_has_sufficient_coverage')
            raw_sides={side:(passes(v['union_all_detections']) or (v['best_single'] is not None and passes(v['best_single']))) for side,v in raw_stats.items()}
            if sum(raw_sides.values())==1:flags_here.append('sufficient_raw_coverage_'+next(side for side,ok in raw_sides.items() if ok)+'_only')
            elif all(raw_sides.values()):flags_here.append('sufficient_raw_coverage_both_sides')
            if gt.get('fallback'):flags_here.append('gt_fallback')
            if 'entirely_outside_viewport' in flags_here:category='gt_outside_image'
            elif 'changed_prediction_or_group_intersects_gt' in flags_here:category='changed_overlap_below_threshold'
            elif ub and passes(ub):category='uncertain_valid_change'
            elif mb and passes(mb):category='matched_below_change_threshold'
            elif sum(raw_sides.values())==1:category='one_sided_detection'
            elif raw_has_union or raw_has_single:category='detection_without_correspondence'
            else:category='detection_insufficient'
            record={'pair_id':pid,'gt_index':target['gt_index'],'original_gt_index':target['original_gt_index'],'gt':gt,'category':category,'category_label':LABELS[category],'flags':flags_here,'visibility':visibility,'best_changed_prediction':cb,'best_group':gg,'best_uncertain_changed':ub,'best_unchanged_matched':mb,'raw_detection_coverage':raw_stats,'root_cause_status':'automated evidence only; semantic identity not human-verified'}
            targets.append(record);sample_targets.append(record);flags.update(flags_here)
        samples.append({'pair_id':pid,'gt_count':fail['gt_count'],'covered':fail['covered'],'missed':fail['missed'],'categories':dict(Counter(t['category'] for t in sample_targets)),'targets':sample_targets})
        byidx={t['original_gt_index']:t for t in sample_targets}
        viewer.append({'id':pid,'summary':dict(fail,diagnostic_categories=[LABELS[k] for k in Counter(t['category'] for t in sample_targets)]),'before':os.path.relpath(row['images']['before'],OUT),'after':os.path.relpath(row['images']['after'],OUT),'gt':[dict(annotations[c['original_gt_index']],original_index=c['original_gt_index'],covered=c['covered'],diagnostic=byidx.get(c['original_gt_index'])) for c in ev['gt_coverage']],
                       'predictions':[dict(before=m['before_bbox'],after=m['after_bbox'],envelope=pipeline.prediction_envelope(m['before_bbox'],m['after_bbox'])) for m in changed]})
    counts=Counter(t['category'] for t in targets);assert len(samples)==60 and len(targets)==108
    for p in protected:assert sha(Path(p['path']))==p['sha256']
    summary={'fail_samples':len(samples),'missed_gt':len(targets),'exclusive_conditions':dict(counts),'secondary_flags_nonexclusive':dict(flags),'note':'Conditions explain where coverage stopped, not verified semantic root causes. Raw detection union is a diagnostic upper bound, never a permitted prediction group.'}
    for name,obj in [('summary.json',summary),('gt_failures.json',targets),('sample_failures.json',samples),('validation.json',{'all_60_failed_samples_analyzed':True,'all_108_missed_gt_analyzed':True,'all_changed_envelopes_and_frozen_groups_confirmed_below_threshold':True,'protected_inputs_unchanged':len(protected)})]:save(OUT/name,obj)
    with (OUT/'gt_failures.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['pair_id','original_gt_index','selector','category','best_prediction_iou','best_prediction_ccr','best_group_iou','best_group_ccr']);w.writeheader()
        for t in targets:w.writerow({'pair_id':t['pair_id'],'original_gt_index':t['original_gt_index'],'selector':t['gt'].get('selector'),'category':t['category'],'best_prediction_iou':(t['best_changed_prediction'] or {}).get('iou'),'best_prediction_ccr':(t['best_changed_prediction'] or {}).get('ccr'),'best_group_iou':(t['best_group'] or {}).get('iou'),'best_group_ccr':(t['best_group'] or {}).get('ccr')})
    template=(ROOT/'geometry/scripts/dataset2_same_gt_review.html').read_text(encoding='utf-8');(OUT/'index.html').write_text(template.replace('/*PAYLOAD*/[]',json.dumps(viewer,ensure_ascii=False).replace('<','\\u003c')).replace('165 mẫu geometry · 421 vùng GT chuẩn.','60 mẫu FAIL · chọn GT missed để xem bằng chứng chẩn đoán.'), encoding='utf-8')
    lines=['# Phân tích FAIL — dataset2, GT chuẩn','','Phạm vi: 60 mẫu FAIL, 108 GT missed trong run 313/421. Không thay thuật toán, tham số, GT hoặc prediction.','','Đây là phân loại điều kiện chấm điểm, không phải nhãn nguyên nhân ngữ nghĩa đã review. Nếu prediction khác giao GT nhưng bbox sai, có thể do detection, correspondence hoặc biểu diễn vùng; không tự coi là matcher đúng.','','## Điều kiện loại trừ nhau','','| Điều kiện | GT | Tỷ lệ | ID mẫu |','|---|---:|---:|---|']
    for k,n in counts.most_common():lines.append(f"| {LABELS[k]} | {n} | {100*n/108:.1f}% | "+', '.join(sorted({t['pair_id'] for t in targets if t['category']==k},key=int))+' |')
    lines+=['','Ưu tiên phân loại: GT hoàn toàn ngoài ảnh → prediction/nhóm thay đổi giao GT nhưng dưới ngưỡng → uncertain thay đổi đạt overlap → matched chưa vượt ngưỡng geometry đạt overlap → detection đủ coverage chỉ một phía → detection đủ coverage cả hai phía nhưng thiếu correspondence → detection thiếu coverage. Một mẫu có thể có nhiều loại GT fail.','','## Dấu hiệu bổ sung — có thể trùng nhau','','```json',json.dumps(dict(flags),indent=2),'```','','Matched chưa thay đổi bao GT có thể là container lớn hoặc phần tử khác, không chứng minh cần hạ ngưỡng classifier. Union toàn bộ detection chỉ cho biết coverage tiềm năng, không được dùng cứu điểm. GT ngoài viewport được giữ nguyên như run chuẩn; đây là dấu hiệu cần review nhãn/ảnh, không tự loại GT.','','## Quan sát ảnh đại diện','','- 548: GT bao cả thanh menu, prediction thay đổi tốt nhất chỉ phủ khoảng 10% GT; cần review độ hạt detection/nhóm và correspondence, chưa thể quy thành lỗi bbox đơn thuần.','- 566: GT bao vùng ô tìm kiếm, prediction thay đổi tốt nhất nằm ở chữ nhỏ lân cận và chỉ phủ khoảng 11,2% GT; điều này không chứng minh phần tử mục tiêu đã được match đúng.','- 606: cặp uncertain có envelope phủ GT, global stability gap 0,0196 dưới ngưỡng 0,05; vẫn cần xác nhận đúng danh tính cặp trước khi chấp nhận.','- 589/GT index 2: matched ứng viên có center shift 2,06px, cạnh tối đa 3px, resize tối đa 4,26%, dưới ngưỡng classifier. Đây là ứng viên theo overlap, chưa phải correspondence được review đúng.','- 521: vùng chữ sát đáy ảnh before, vùng tương ứng sau thay đổi không còn chữ trong crop; detection đủ coverage chỉ phía before. Đây là ví dụ cần xem clipping/unmatched thay vì chỉ chỉnh overlap.','','Ảnh minh họa: `examples.jpg` (đỏ GT, xanh một ứng viên tương ứng theo diagnostic; không phải xác nhận cặp đúng).','','## Từng mẫu','','| ID | Covered/GT | GT thiếu | Điều kiện |','|---|---:|---:|---|']
    lines += [f"| {s['pair_id']} | {s['covered']}/{s['gt_count']} | {s['missed']} | "+'; '.join(f'{LABELS[k]}: {v}' for k,v in s['categories'].items())+' |' for s in samples]
    lines+=['','Chi tiết số IoU/CCR, ID cặp, bbox, delta và flags: `gt_failures.json`. Bảng gọn: `gt_failures.csv`. Viewer: `index.html`, chọn GT missed để xem diagnostic.','', 'Tái lập: `python3 -m geometry.scripts.analyze_dataset2_failures`.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
