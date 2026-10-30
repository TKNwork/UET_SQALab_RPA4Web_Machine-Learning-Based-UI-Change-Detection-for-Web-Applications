"""Frozen geometry pipeline on fresh dataset2 detections, with GT completeness accounting."""
import gzip
import csv
import json
import os
import time
from pathlib import Path
from collections import Counter
from PIL import Image
from geometry import pipeline
from geometry.prediction_groups import build_prediction_groups
from geometry.scripts.detect_dataset2 import ROOT,save,sha
OUT=ROOT/'geometry/results/dataset2_from502_detection_v1'
DET=ROOT/'dataset2_detection'

def load_result(pid):
    with gzip.open(OUT/pid/'matching_result_full.json.gz','rt') as f:return json.load(f)

def evaluate(rows,tag):
    results=[]
    for row in rows:
        pid=row['pair_id'];dest=OUT/pid;gt=Path(row['gt_path']);r=load_result(pid)
        if row['gt_count']==0:
            groups=json.loads((dest/'groups_edges8.json').read_text(encoding='utf-8'))
            results.append(dict(row,status='not_scored_unresolved_gt',covered=0,summary=r['summary'],predictions=groups['selected_predictions'],groups=len(groups['groups'])));continue
        groups=json.loads((dest/'groups_edges8.json').read_text(encoding='utf-8'))
        for mode in ('single_envelope','union_support'):
            ev=pipeline.evaluate(r['matched_pairs'],gt,'edges8') if mode=='single_envelope' else pipeline.evaluate_grouped(r['matched_pairs'],gt,'edges8',groups,mode)
            assert ev['gt_count']==row['gt_count'] and ev['gt_count']>0
            ev['fully_resolved_gt']=row['fully_resolved'];ev['subset_pass']=ev['pass'];ev['pass']=ev['pass'] if row['fully_resolved'] else None
            save(dest/f'eval_edges8_{mode}.json',ev)
        results.append(dict(row,status='complete' if row['fully_resolved'] else 'partial_gt',covered=ev['covered'],missed=ev['missed_gt'],pass_=ev['pass'],predictions=ev['predictions'],groups=ev['group_count'],summary=r['summary']))
    scored=[r for r in results if 'pass_' in r];complete=[r for r in scored if r['fully_resolved']]
    summary={'samples_requested':len(rows),'samples_scored':len(scored),'fully_scored_samples':len(complete),'partial_gt_samples':sum(not r['fully_resolved'] for r in scored),'not_scored_unresolved_gt':sum(r['status']=='not_scored_unresolved_gt' for r in results),'gt_count':sum(r['gt_count'] for r in scored),'covered':sum(r['covered'] for r in scored),'pass_samples':sum(r['pass_'] is True for r in complete),'fail_samples':sum(r['pass_'] is False for r in complete),'coverage':sum(r['covered'] for r in scored)/max(1,sum(r['gt_count'] for r in scored))}
    summary.update(fully_scored_gt=sum(r['gt_count'] for r in complete),fully_scored_covered=sum(r['covered'] for r in complete),partially_scored_gt=sum(r['gt_count'] for r in scored if not r['fully_resolved']),partially_scored_covered=sum(r['covered'] for r in scored if not r['fully_resolved']),prediction_count=sum(r['predictions'] for r in scored),group_count=sum(r['groups'] for r in scored))
    summary['all_sample_prediction_count']=sum(r['predictions'] for r in results)
    summary['all_sample_group_count']=sum(r['groups'] for r in results)
    summary['fully_scored_coverage']=summary['fully_scored_covered']/max(1,summary['fully_scored_gt'])
    summary['matching_totals']={k:sum(r['summary'][k] for r in results) for k in ('matched_pairs','uncertain_pairs','unmatched_before','unmatched_after')}
    save(OUT/f'{tag}_results.json',{'summary':summary,'samples':results})
    return summary,results

def main():
    manifest=json.loads((OUT/'gt_manifest.json').read_text(encoding='utf-8'));rows=manifest['samples']
    config=json.loads((ROOT/'geometry/results/dataset2_from502_v1/frozen_config.json').read_text(encoding='utf-8'))
    cfg=dict(config['matching']);cfg.pop('image_root',None)
    save(OUT/'frozen_config.json',dict(config,primary_classifier='edges8',primary_mode='union_support'))
    pending=list(rows);ready=[];failed=[];smoke=False
    first5={r['pair_id'] for r in rows[:5]}
    while pending:
        changed=False
        for row in list(pending):
            pid=row['pair_id'];folder=Path(row['folder']);inputs=[]
            for side in ('baseline','inject'):
                receipt=DET/side/'receipts'/f'{folder.name}.json';det=DET/side/'json'/f'{folder.name}_full_hierarchy.json'
                if not receipt.exists() or not det.exists():break
                rec=json.loads(receipt.read_text(encoding='utf-8'))
                if sha(det)!=rec['output_sha256'] or sha(folder/('baseline.png' if side=='baseline' else 'inject.png'))!=rec['image_sha256']:raise ValueError('Detector input provenance mismatch')
                inputs.append(det)
            if len(inputs)!=2:continue
            try:
                pair=OUT/'pairs'/pid;pair.mkdir(parents=True,exist_ok=True)
                for source,name in [(inputs[0],'before.json'),(inputs[1],'after.json'),(folder/'baseline.png','before.png'),(folder/'inject.png','after.png')]:
                    target=pair/name
                    if not target.exists():target.symlink_to(source)
                dest=OUT/pid;dest.mkdir(exist_ok=True)
                provenance={'inputs':{str(p):sha(p) for p in inputs+[folder/'baseline.png',folder/'inject.png']},'config':cfg,'pipeline_sha256':sha(ROOT/'geometry/pipeline.py'),'shared_sha256':sha(ROOT/'shared/matching.py'),'group_sha256':sha(ROOT/'geometry/prediction_groups.py')}
                frozen=dest/'prediction_provenance.json'
                if frozen.exists():
                    assert json.loads(frozen.read_text(encoding='utf-8'))==provenance
                    r=load_result(pid)
                else:
                    r=pipeline.process_pair(pair,OUT,cfg)
                    assert r['summary']['crop_available']
                    with gzip.open(dest/'matching_result_full.json.gz','wt',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,separators=(',',':'))
                    (dest/'matching_result_full.json').unlink()
                    with Image.open(folder/'baseline.png') as im:size=im.size
                    selected=[m for m in r['matched_pairs'] if pipeline.classify_change(m['delta'],'edges8')]
                    groups=build_prediction_groups(selected,r['matched_pairs'],size,config['grouping'])
                    save(dest/'groups_edges8.json',groups);save(frozen,provenance)
                ready.append(row);pending.remove(row);changed=True
                print(f'Matching/group frozen {pid}: {len(ready)}/{len(rows)}',flush=True)
                save(OUT/'run_status.json',{'status':'matching_and_grouping_without_gt','completed_predictions':[r['pair_id'] for r in ready],'pending':[r['pair_id'] for r in pending],'failed':failed})
                if not smoke and first5.issubset({r['pair_id'] for r in ready}):
                    summary,_=evaluate(rows[:5],'smoke5');smoke=True;print('Smoke5 '+json.dumps(summary),flush=True)
            except Exception as exc:
                failed.append({'pair_id':pid,'error':type(exc).__name__+': '+str(exc)});pending.remove(row);changed=True
        if pending:
            ds=json.loads((DET/'run_status.json').read_text(encoding='utf-8')) if (DET/'run_status.json').exists() else {}
            if ds.get('status') in ('complete','partial_or_blocked'):
                failed.extend({'pair_id':r['pair_id'],'error':'Missing successful detector receipts'} for r in pending);pending=[];break
            if not changed:time.sleep(5)
    # Main scoring starts after all obtainable predictions/groups are frozen.
    summary,results=evaluate(sorted(ready,key=lambda r:int(r['pair_id'])),'all')
    save(OUT/'summary.json',summary)
    save(OUT/'failure_samples.json',[r for r in results if r.get('pass_') is not True])
    with (OUT/'evaluation.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['pair_id','status','fully_resolved','gt_count','covered','pass_']);writer.writeheader();writer.writerows({k:r.get(k) for k in writer.fieldnames} for r in results)
    save(OUT/'run_manifest.json',{'selected_ids':[r['pair_id'] for r in rows],'prediction_ids':[r['pair_id'] for r in ready],'failed':failed,'main_scoring_after_all_groups_frozen':True,'smoke5_scored_early':smoke,'gt_manifest_sha256':sha(OUT/'gt_manifest.json'),'summary':summary})
    save(OUT/'run_status.json',{'status':'complete_with_gt_limitations' if not failed else 'partial','summary':summary,'failed':failed})
    report(results,summary,failed,manifest)
    print(json.dumps(summary,indent=2),flush=True)

def report(results,summary,failed,manifest):
    old=json.loads((ROOT/'geometry/results/group_coverage_v1/summary.json').read_text(encoding='utf-8'))['edges8']['union_support']
    lines=['# Dataset2 geometry — edges8 + union_support','','## Phạm vi và giới hạn','','Detection: toàn bộ dataset2; matching/scoring: ID ≥ 502 có nhãn geometry. Dùng current dataset1 detector entrypoint; không có hash upstream lịch sử để chứng minh detector bit-for-bit tương đương. Matching/grouping giữ cấu hình chuẩn.','','GT mới là bbox DOM hai endpoint, GT cũ là vùng pixel-diff tight theo README dataset gốc. Đã tạo envelope từ endpoint xác định được; KHÔNG khẳng định tương đương protocol nhãn cũ. Không ghép selector lặp theo thứ tự; dùng selector DOM duy nhất và bbox sai số dưới 1px. Trường hợp mơ hồ còn unresolved và không được tính PASS đầy đủ.','','## Kết quả trên phần GT đã chuẩn hóa','','```json',json.dumps(summary,ensure_ascii=False,indent=2),'```','','## Chuẩn hóa GT','','```json',json.dumps(manifest['summary'],ensure_ascii=False,indent=2),'```','','## Đối chiếu tập cũ','','Tập cũ: '+str(old['covered'])+'/'+str(old['gt_count'])+' GT, '+str(old['pass_samples'])+'/'+str(old['samples'])+' mẫu PASS. Tập mới khác mẫu và khác nguồn biểu diễn GT; không diễn giải chênh lệch là cải thiện/suy giảm thuật toán. Coverage không phải precision correspondence.','','## Lỗi vận hành','','```json',json.dumps(failed,ensure_ascii=False,indent=2),'```','','## Tái lập','','`python3 -m geometry.scripts.detect_dataset2`','','`python3 -m geometry.scripts.normalize_dataset2_gt`','','`python3 -m geometry.scripts.evaluate_dataset2`','','## Từng mẫu','','| ID | GT đã chuẩn hóa | Phủ | Trạng thái |','|---|---:|---:|---|']
    lines += [f"| {r['pair_id']} | {r['gt_count']} | {r.get('covered',0)} | {r['status']} / PASS={r.get('pass_')} |" for r in results]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Self-contained GT/prediction viewer with relative image paths; no network fetch.
    payload=[]
    for row in results:
        pid=row['pair_id'];r=load_result(pid);g=json.loads(Path(row['gt_path']).read_text(encoding='utf-8'));groups=json.loads((OUT/pid/'groups_edges8.json').read_text(encoding='utf-8'))
        payload.append({'id':pid,'status':row['status'],'summary':row,'before':os.path.relpath(Path(row['folder'])/'baseline.png',OUT),'after':os.path.relpath(Path(row['folder'])/'inject.png',OUT),'gt':g,'raw_gt':json.loads((ROOT/'groundtruth/groundtruth'/f'{int(pid):03d}.json').read_text(encoding='utf-8')),'predictions':[{'before':m['before_bbox'],'after':m['after_bbox']} for m in r['matched_pairs'] if pipeline.classify_change(m['delta'],'edges8')],'groups':[{'bbox':x['prediction_region'],'members':x['member_regions']} for x in groups['groups']]})
    html='''<!doctype html><meta charset="utf-8"><title>Dataset2 geometry review</title><style>body{font:15px sans-serif;margin:20px;background:#151922;color:#eee}select,button{padding:8px}main{display:flex;gap:12px}.pane{width:49%}svg{width:100%;background:white}pre{white-space:pre-wrap}</style><h1>Dataset2 geometry review</h1><p>Đỏ: GT envelope chuẩn hóa. Xanh: prediction ở từng phía. Cam: bbox nhóm (không đồng nghĩa vùng union support). GT unresolved không được tính PASS.</p><select id="pick"></select><label><input id="pred" type="checkbox" checked>Prediction</label><label><input id="raw" type="checkbox">GT endpoint gốc (tím)</label><label><input id="grp" type="checkbox">Nhóm</label><main><div class="pane"><h3>Before</h3><svg id="before"></svg></div><div class="pane"><h3>After</h3><svg id="after"></svg></div></main><pre id="info"></pre><script>const data=PAYLOAD;let pick=document.getElementById('pick');data.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=s.id+' — '+s.summary.covered+'/'+s.summary.gt_count+' — '+s.status+' — PASS='+s.summary.pass_;pick.append(o)});function rect(b,color){return `<rect x="${b[0]}" y="${b[1]}" width="${b[2]-b[0]}" height="${b[3]-b[1]}" fill="none" stroke="${color}" stroke-width="2"/>`}function render(){let s=data[+pick.value];document.getElementById('info').textContent=JSON.stringify({summary:s.summary,unresolved:s.gt.unresolved},null,2);['before','after'].forEach(side=>{let svg=document.getElementById(side),img=new Image();img.onload=()=>{svg.setAttribute('viewBox',`0 0 ${img.width} ${img.height}`);svg.innerHTML=`<image href="${s[side]}" width="${img.width}" height="${img.height}"/>`+s.gt.bboxes.map(b=>rect([b.x,b.y,b.x+b.w,b.y+b.h],'red')).join('')+(document.getElementById('raw').checked?s.raw_gt[side+'_bboxes'].filter(b=>b.type.includes('geometry')).map(b=>rect([b.x,b.y,b.x+b.w,b.y+b.h],'#c000ff')).join(''):'')+(document.getElementById('pred').checked?s.predictions.map(p=>rect(p[side],'#00aafa')).join(''):'')+(document.getElementById('grp').checked?s.groups.map(g=>rect(g.bbox,'orange')).join(''):'')};img.src=s[side]})}pick.onchange=render;document.getElementById('raw').onchange=render;document.getElementById('pred').onchange=render;document.getElementById('grp').onchange=render;render();</script>'''
    (OUT/'index.html').write_text(html.replace('PAYLOAD',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')), encoding='utf-8')
if __name__=='__main__':main()
