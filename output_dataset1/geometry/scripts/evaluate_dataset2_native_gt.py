"""Score all dataset2 geometry annotations directly, preserving old partial run."""
import csv
import gzip
import json
import os
from pathlib import Path
from collections import Counter
from geometry.scripts.detect_dataset2 import ROOT,save,sha
from geometry.native_gt_evaluation import evaluate_native,PROTOCOL,THRESHOLDS
from geometry.pipeline import classify_change,prediction_envelope,evaluate,evaluate_grouped
SOURCE=ROOT/'geometry/results/dataset2_from502_detection_v1'
OUT=ROOT/'geometry/results/dataset2_native_gt_v1'

def summarize(rows):
    count=sum(r['gt_count'] for r in rows);covered=sum(r['covered'] for r in rows)
    return {'samples':len(rows),'gt_endpoint_count':count,'covered':covered,'missed':count-covered,'coverage':covered/count if count else None,
            'pass_samples':sum(r['pass'] for r in rows),'fail_samples':sum(not r['pass'] for r in rows),
            'per_side':{s:{'gt_count':sum(r['per_side'][s]['gt_count'] for r in rows),'covered':sum(r['per_side'][s]['covered'] for r in rows)} for s in ('before','after')}}

def main():
    OUT.mkdir(exist_ok=True)
    rows=json.loads((SOURCE/'gt_manifest.json').read_text(encoding='utf-8'))['samples']
    # Freeze every prediction/group hash before any scoring in this run.
    protected=[]
    for row in rows:
        pid=row['pair_id']
        for name in ('matching_result_full.json.gz','groups_edges8.json','prediction_provenance.json'):
            p=SOURCE/pid/name;protected.append({'path':str(p),'sha256':sha(p)})
    for name in ('summary.json','all_results.json','gt_manifest.json','REPORT.md','index.html'):
        p=SOURCE/name;protected.append({'path':str(p),'sha256':sha(p)})
    save(OUT/'frozen_prediction_manifest.json',{'phase':'all_predictions_frozen_before_scoring','inputs':protected})
    results=[];payload=[];inputs=[]
    for row in rows:
        pid=row['pair_id'];folder=Path(row['folder']);gp=ROOT/'groundtruth/groundtruth'/f'{int(pid):03d}.json'
        gt=json.loads(gp.read_text(encoding='utf-8'));assert str(int(gt['pair_id']))==str(int(pid)) and int(pid)>=502
        inputs.append({'path':str(gp),'sha256':sha(gp)})
        with gzip.open(SOURCE/pid/'matching_result_full.json.gz','rt') as f:r=json.load(f)
        groups=json.loads((SOURCE/pid/'groups_edges8.json').read_text(encoding='utf-8'))
        for mode in ('single_envelope','union_support'):
            ev=evaluate_native(r['matched_pairs'],gt,groups,'edges8',mode)
            # Verify numerical parity with the unchanged legacy evaluator on the same native annotations.
            adapter=OUT/pid/'native_gt_as_regions.json'
            save(adapter,{'pair_id':pid,'bboxes':[dict(b,source_side=side,source_index=i) for side in ('before','after') for i,b in enumerate(gt[side+'_bboxes']) if 'geometry' in b.get('type',[])]})
            reference=evaluate(r['matched_pairs'],adapter,'edges8') if mode=='single_envelope' else evaluate_grouped(r['matched_pairs'],adapter,'edges8',groups,'union_support')
            assert all(ev[k]==reference[k] for k in ('gt_count','covered','pass','predictions','supporting_predictions','unverified'))
            ev['unchanged_legacy_evaluator_parity_verified']=True
            save(OUT/pid/f'eval_edges8_{mode}.json',ev)
            if mode=='single_envelope':single=ev
        assert ev['covered']>=single['covered']
        result={'pair_id':pid,'previously_unresolved':not row['fully_resolved'],'gt_count':ev['gt_count'],'covered':ev['covered'],'single_covered':single['covered'],
                'group_rescued':ev['covered']-single['covered'],'pass':ev['pass'],'per_side':ev['per_side'],'missed_gt':ev['missed_gt'],'predictions':ev['predictions'],'groups':ev['groups']}
        results.append(result)
        payload.append({'id':pid,'previously_unresolved':result['previously_unresolved'],'summary':result,'before':os.path.relpath(folder/'baseline.png',OUT),'after':os.path.relpath(folder/'inject.png',OUT),
                        'gt':gt,'coverage':ev['gt_coverage'],'predictions':[{'before':m['before_bbox'],'after':m['after_bbox'],'envelope':prediction_envelope(m['before_bbox'],m['after_bbox'])} for m in r['matched_pairs'] if classify_change(m['delta'],'edges8')],
                        'groups':[{'id':g['group_id'],'envelopes':g['member_regions'],'before':[m['before_bbox'] for m in g['members']],'after':[m['after_bbox'] for m in g['members']]} for g in groups['groups']]})
    summary={'protocol':PROTOCOL,'thresholds':THRESHOLDS,'primary_mode':'union_support','all':summarize(results),
             'previously_unresolved_31':summarize([r for r in results if r['previously_unresolved']]),
             'previously_resolved_135':summarize([r for r in results if not r['previously_unresolved']]),
             'single_envelope':{'covered':sum(r['single_covered'] for r in results)},'group_rescued_endpoint_count':sum(r['group_rescued'] for r in results),
             'prediction_count':sum(r['predictions'] for r in results),'group_count':sum(r['groups'] for r in results)}
    frozen=protected+inputs
    assert all(sha(Path(e['path']))==e['sha256'] for e in frozen)
    assert len(results)==166 and summary['all']['gt_endpoint_count']==799
    protected_old=json.loads((ROOT/'geometry/results/dataset2_from502_v1/protected_inputs.json').read_text(encoding='utf-8'))
    assert all(sha(Path(e['path']))==e['sha256'] for e in protected_old)
    for name,obj in [('summary.json',summary),('all_results.json',results),('failure_samples.json',[r for r in results if not r['pass']]),('previously_unresolved_31.json',[r for r in results if r['previously_unresolved']])]:save(OUT/name,obj)
    with (OUT/'evaluation.csv').open('w') as f:
        keys=['pair_id','previously_unresolved','gt_count','covered','single_covered','group_rescued','pass','predictions','groups'];writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in results)
    save(OUT/'run_manifest.json',{'protocol':PROTOCOL,'inputs':frozen,'code_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'geometry/native_gt_evaluation.py',Path(__file__),ROOT/'geometry/pipeline.py',ROOT/'shared/matching.py',ROOT/'geometry/prediction_groups.py']},'predictions_regenerated':False,'config':json.loads((SOURCE/'frozen_config.json').read_text(encoding='utf-8')),'gt_identity_pairing_used':False})
    save(OUT/'validation.json',{'all_166_samples_scored':True,'all_799_original_geometry_annotations_scored':True,'before_annotations':summary['all']['per_side']['before']['gt_count'],'after_annotations':summary['all']['per_side']['after']['gt_count'],'no_gt_conversion_or_dropped_annotations':True,'frozen_predictions_and_previous_report_unchanged':True,'protected_original_files_verified':len(protected_old),'unchanged_legacy_evaluator_scores_equal_on_all_166_samples':True,'human_semantic_review_complete':False})
    report(summary,results)
    viewer(payload)
    save(OUT/'run_status.json',{'status':'complete','summary':summary})
    print(json.dumps(summary,indent=2))

def report(summary,results):
    lines=['# Dataset2 — đánh giá geometry trực tiếp trên GT gốc','','## Thay đổi ở evaluator','','Đã đối chiếu điểm với evaluator gốc trên cùng danh sách GT gốc ở từng phía: gt_count, covered, PASS, predictions, supporting_predictions và unverified trùng khớp trên cả 166 mẫu, ở cả hai mode. Matcher, classifier edges8 và frozen groups không thay đổi; không chạy lại detector/matching. Bổ sung protocol `geometry_native_gt_envelope_coverage_v1` cho schema GT hai phía. Tất cả annotation geometry gốc được chấm, không cần chuyển GT thành envelope hoặc suy đoán danh tính endpoint.','','Chấm riêng từng bbox GT gốc ở before và after, nhưng giữ nguyên vùng prediction như evaluator cũ: envelope bao bbox prediction before và after; nhóm dùng union chính xác của các envelope thành viên. Không suy đoán cặp GT và không dùng GT để tạo vùng prediction. Đạt nếu IoU ≥ 0.30 HOẶC CCR ≥ 0.50. Diện tích bên trong envelope mỗi cặp vẫn được tính như evaluator cũ; khoảng trống giữa các envelope thành viên không được tô kín thành bbox bao nhóm. Link chỉ có envelope phủ GT nhưng endpoint cùng phía không giao được gắn cờ để review. Một GT có thể có nhiều bằng chứng; không assignment one-to-one với GT. Không gom tùy ý mọi prediction trong ảnh.','','PASS một mẫu đòi hỏi tất cả GT geometry gốc ở cả hai phía được phủ. Annotation có selector lặp, tọa độ trùng, khác số lượng hai phía và ngoài viewport vẫn giữ nguyên index và được chấm. Một phía không có GT không làm rỗng phía còn lại; cả hai phía không có geometry bị báo lỗi thay vì PASS.','','## Đơn vị và giới hạn so sánh','','Đơn vị mới là **bbox annotation ở một phía**, không phải một đối tượng before–after hay một GT envelope. 409 bbox before + 390 bbox after = 799 annotation; không gọi đây là 799 thay đổi độc lập. Protocol được áp dụng đồng nhất cho cả 166 mẫu, không chỉ vá 31 mẫu.','','Số liệu không so trực tiếp với 212/298 (71,1%) của lần chấm envelope một phần hay 220/270 của tập cũ. Đây là thay đổi cách biểu diễn/chấm GT, không phải cải thiện matcher. Coverage không xác nhận độ chính xác correspondence. Báo cáo cũ và các GT envelope chuyển đổi được giữ nguyên.','','## Kết quả','','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','','## Từng mẫu','','| ID | Từng vướng chuẩn hóa | GT endpoint | Phủ | PASS |','|---|---|---:|---:|---|']
    lines += [f"| {r['pair_id']} | {r['previously_unresolved']} | {r['gt_count']} | {r['covered']} | {r['pass']} |" for r in results]
    lines += ['','## Tái lập','','`python3 -m geometry.scripts.evaluate_dataset2_native_gt`','','Viewer: `index.html`. Chi tiết từng GT và link bằng chứng: `<ID>/eval_edges8_union_support.json`. Danh sách 31 mẫu đã chấm: `previously_unresolved_31.json`.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

def viewer(payload):
    html='''<!doctype html><meta charset="utf-8"><title>Geometry — native GT</title><style>body{font:15px sans-serif;background:#151922;color:#eee;margin:20px}select,button{padding:8px}main{display:flex;gap:10px}.pane{width:49%}svg{width:100%;background:white}pre{white-space:pre-wrap}</style><h1>Geometry — GT gốc ở từng ảnh</h1><p>GT xanh lá: covered; đỏ: missed. Prediction endpoint xanh dương; envelope nét đứt cam. Đơn vị: annotation ở từng phía, không phải GT envelope.</p><label><input id="only31" type="checkbox">Chỉ 31 mẫu từng vướng GT</label><label><input id="fail" type="checkbox">Chỉ mẫu FAIL</label><select id="pick"></select><label><input id="pred" type="checkbox" checked>Prediction</label><label><input id="grp" type="checkbox">Union thành viên nhóm</label><main><div class="pane"><h3>Before</h3><svg id="before"></svg></div><div class="pane"><h3>After</h3><svg id="after"></svg></div></main><pre id="info"></pre><script>const data=PAYLOAD;const $=id=>document.getElementById(id);function rect(b,color,label){return `<rect x="${b[0]}" y="${b[1]}" width="${b[2]-b[0]}" height="${b[3]-b[1]}" fill="none" stroke="${color}" stroke-width="2"><title>${label||''}</title></rect>`}function options(){const old=$('pick').value;$('pick').replaceChildren();data.forEach((s,i)=>{if($('only31').checked&&!s.previously_unresolved||$('fail').checked&&s.summary.pass)return;let o=document.createElement('option');o.value=i;o.textContent=s.id+' '+s.summary.covered+'/'+s.summary.gt_count+' '+(s.summary.pass?'PASS':'FAIL');$('pick').append(o)});if([...$('pick').options].some(o=>o.value===old))$('pick').value=old;render()}function render(){if(!$('pick').options.length){$('info').textContent='Không có mẫu';$('before').replaceChildren();$('after').replaceChildren();return}let s=data[+$('pick').value];$('info').textContent=JSON.stringify(s.summary,null,2);['before','after'].forEach(side=>{const image=new Image();image.onload=()=>{if(data[+$('pick').value]!==s)return;let svg=$(side);svg.setAttribute('viewBox',`0 0 ${image.width} ${image.height}`);svg.innerHTML=`<image href="${s[side]}" width="${image.width}" height="${image.height}"/>`+($('pred').checked?s.predictions.map(p=>rect(p[side],'#008dff','prediction')+rect(p.envelope,'#ffad33','envelope').replace('<rect','<rect stroke-dasharray="5 3"')).join(''):'')+($('grp').checked?s.groups.flatMap(g=>g.envelopes.map(b=>rect(b,'orange',g.id))).join(''):'')+s.coverage.filter(g=>g.side===side).map(g=>rect([g.gt.x,g.gt.y,g.gt.x+g.gt.w,g.gt.y+g.gt.h],g.covered?'#00e46c':'red',side+' GT'+g.original_gt_index)).join('')};image.src=s[side]})}['only31','fail'].forEach(id=>$(id).onchange=options);['pick','pred','grp'].forEach(id=>$(id).onchange=render);options();</script>'''
    (OUT/'index.html').write_text(html.replace('PAYLOAD',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')), encoding='utf-8')

if __name__=='__main__':main()
