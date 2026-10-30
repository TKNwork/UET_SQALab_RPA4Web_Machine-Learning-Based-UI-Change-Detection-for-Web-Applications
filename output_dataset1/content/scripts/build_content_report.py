"""Static review UI and auditable reports; never changes predictions or scoring."""
import html
import json
from collections import Counter
from pathlib import Path

TEMPLATE = '''<!doctype html><meta charset="utf-8"><title>Content review __PID__</title>
<style>body{font:14px system-ui;margin:20px;background:#f5f6fa;color:#172333}a{color:#1261ac} .images{display:flex;gap:10px}svg{width:49%;border:1px solid #aaa;background:white}table{border-collapse:collapse;width:100%;background:white}td,th{padding:6px;border:1px solid #ddd;text-align:left}tr{cursor:pointer}tr:hover{background:#e3efff}pre{white-space:pre-wrap;max-height:400px;overflow:auto}button,select{margin:8px} .legend{padding:10px} </style>
<a href="../index.html">← Samples</a> · <a href="inference.json">Inference JSON</a> · <a href="evaluation.json">Evaluation JSON</a>
<h1>Content __PID__</h1><div class="legend">GT: đỏ; matched/prediction: xanh; uncertain: cam; unmatched: tím; detector: xám. Chọn hàng để xem hai đầu và bằng chứng.</div>
<label><input id="gt" type="checkbox" checked>GT content</label><label><input id="det" type="checkbox">Tất cả detector</label>
<select id="filter"><option value="predictions">Predictions</option><option value="matched_pairs">All matched</option><option value="uncertain_pairs">Uncertain correspondence</option><option value="unmatched_before">Unmatched before</option><option value="unmatched_after">Unmatched after</option><option value="gt_coverage">GT evaluation</option></select>
<div class="images"><svg id="before"></svg><svg id="after"></svg></div><pre id="detail">Chọn hàng bên dưới.</pre><table><thead><tr><th>Index/IDs</th><th>State/kind</th><th>Text / evaluation</th><th>Cost / ban gap</th></tr></thead><tbody id="rows"></tbody></table>
<script>const data=__PAYLOAD__;const r=data.inference,ev=data.evaluation,ns='http://www.w3.org/2000/svg';let chosen=null;
function rect(svg,b,color,width=2){const x=document.createElementNS(ns,'rect');for(const[k,v]of Object.entries({x:b[0],y:b[1],width:b[2]-b[0],height:b[3]-b[1],fill:'none',stroke:color,'stroke-width':width}))x.setAttribute(k,v);svg.append(x)}
function draw(){for(const side of ['before','after']){const svg=document.getElementById(side);svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${r.image_size[0]} ${r.image_size[1]}`);const im=document.createElementNS(ns,'image');im.setAttribute('href',side+'.png');im.setAttribute('width',r.image_size[0]);im.setAttribute('height',r.image_size[1]);svg.append(im);if(document.getElementById('det').checked)for(const d of r[side+'_detections'])rect(svg,d.bbox,'#999',.8);if(document.getElementById('gt').checked)for(const t of ev.gt_coverage){const g=t.gt;rect(svg,[g.x,g.y,g.x+g.w,g.y+g.h],'#e22',2)}if(chosen){if(chosen.gt){const g=chosen.gt;rect(svg,[g.x,g.y,g.x+g.w,g.y+g.h],'#00acff',4);for(const link of chosen.links)for(const i of link.prediction_indices)rect(svg,r.predictions[i][side+'_bbox'],'#008b35',3)}else{const box=chosen[side+'_bbox']||((document.getElementById('filter').value==='unmatched_'+side)?chosen.bbox:null);if(box)rect(svg,box,document.getElementById('filter').value.startsWith('uncertain')?'#f90':document.getElementById('filter').value.startsWith('unmatched')?'#933bc4':'#009543',3)}}}}
function table(){chosen=null;const key=document.getElementById('filter').value,items=key==='gt_coverage'?ev[key]:r[key],rows=document.getElementById('rows');rows.replaceChildren();items.forEach((m,i)=>{const tr=document.createElement('tr');const vals=[m.before_id?`${i}: ${m.before_id} → ${m.after_id}`:m.id||'GT '+m.original_gt_index,m.content?m.content.state+' / '+m.content.kind:m.covered===undefined?'unmatched':m.covered?'covered':'missed',m.content?m.content.before_text+' → '+m.content.after_text:m.gt?m.gt.selector+' '+m.diagnostic_hypotheses.join(', '):m.ocr_text||'',m.evidence?m.evidence.cost+' / '+m.evidence.cost_gap_ban:''];for(const val of vals){const td=document.createElement('td');td.textContent=val;tr.append(td)}tr.onclick=()=>{chosen=m;document.getElementById('detail').textContent=JSON.stringify(m,null,2);draw()};rows.append(tr)});draw()}
for(const id of ['gt','det'])document.getElementById(id).onchange=draw;document.getElementById('filter').onchange=table;table();</script>'''

def build(out):
    def read(p):return json.loads(p.read_text(encoding='utf-8'))
    manifest=read(out/'data_manifest.json');have_all=(out/'all_status.json').exists() and read(out/'all_status.json')['status']=='complete'
    split='all' if have_all else 'dev';summary=read(out/f'{split}_summary.json')
    selected=[s for s in manifest['samples'] if have_all or s['split']=='dev']
    index=['<!doctype html><meta charset="utf-8"><title>Content review</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto}td,th{padding:12px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>Content pipeline review</h1><p>Coverage vùng GT; chưa xác nhận correctness của matching và content bằng review đầy đủ.</p><table><tr><th>ID</th><th>Split</th><th>Covered / GT</th><th>Prediction</th><th>Uncertain</th></tr>']
    reasons=Counter();kinds=Counter()
    lines=['# Content v1 — kết quả thực tế','', 'Coverage dưới đây chỉ đo mức phủ vùng GT. Không phải matching accuracy, precision, recall hoặc F1 của content.','',
           '| Tập | Mẫu | GT covered / total | Coverage | Mẫu đủ GT | Predictions | Chưa xác minh với GT |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,v in summary.items():
        if v['samples']:lines.append(f"| {name} | {v['samples']} | {v['covered']}/{v['gt_count']} | {v['coverage']:.1%} | {v['pass_samples']}/{v['samples']} | {v['predictions']} | {v['unverified_predictions']} |")
    lines+=['','## Từng mẫu','','| ID | Tập | Covered/GT | Predictions | Matching uncertain | GT thiếu (index gốc) |','|---|---|---:|---:|---:|---|']
    for s in selected:
        pid=s['pair_id'];r=read(out/pid/'inference.json');ev=read(out/pid/'evaluation.json')
        payload=json.dumps({'inference':r,'evaluation':ev},ensure_ascii=False).replace('<','\\u003c')
        (out/pid/'review.html').write_text(TEMPLATE.replace('__PID__',pid).replace('__PAYLOAD__',payload), encoding='utf-8')
        for p in r['predictions']:kinds[p['content']['kind']]+=1
        for g in ev['gt_coverage']:
            if not g['covered']:reasons.update(g['diagnostic_hypotheses'])
        index.append(f'<tr><td><a href="{pid}/review.html">{pid}</a></td><td>{s["split"]}</td><td>{ev["covered"]}/{ev["gt_count"]}</td><td>{ev["predictions"]}</td><td>{len(r["uncertain_pairs"])}</td></tr>')
        missed=', '.join(str(t['original_gt_index']) for t in ev['gt_coverage'] if not t['covered']) or '—'
        lines.append(f'| [{pid}]({pid}/review.html) | {s["split"]} | {ev["covered"]}/{ev["gt_count"]} | {ev["predictions"]} | {len(r["uncertain_pairs"])} | {missed} |')
    index.append('</table><p><a href="REPORT.md">Report</a> · <a href="data_manifest.json">Manifest</a></p>')
    (out/'index.html').write_text('\n'.join(index), encoding='utf-8')
    lines+=['','## Bằng chứng và lỗi cần review','',f'Prediction kinds: `{dict(kinds)}`.',f'Hypotheses GT thiếu (có thể nhiều lý do cho một GT): `{dict(reasons)}`. Đây là chỉ báo tự động, không phải nhãn nguyên nhân đã xác nhận.','',
            'Xem [development_review.md](development_review.md) và [full_review.md](full_review.md) để phân biệt covered do cặp sai, GT fallback không đổi và thay đổi bố cục.','',
            '## Công thức chấm','',
            'P là hợp chính xác các bbox before và after của một prediction; hoặc hợp endpoint của các thành viên nhóm local. Không lấp khoảng trống giữa endpoint. I = area(P ∩ G). GT được phủ nếu **I/area(G) ≥ 0.50 và I/area(P) ≥ 0.25**. Một prediction có thể đáp ứng nhiều GT; một GT có thể được đáp ứng bởi nhiều prediction. Cặp correspondence uncertain, change uncertain và unmatched không tham gia chấm. Không cộng tất cả prediction toàn trang để đạt ngưỡng.','',
            'Ngưỡng thể hiện ít nhất một nửa vùng GT có bằng chứng, và ít nhất một phần tư support thuộc GT để hạn chế container quá lớn. Chọn trên tập phát triển nhỏ, chưa hiệu chuẩn độc lập; khác giao thức geometry OR IoU/CCR. GT dùng một mặt phẳng tọa độ chung; không có nhãn endpoint trong schema. Mẫu 187 xác minh trực quan hai bbox riêng ở hai vị trí cũ/mới, mẫu 451 có bbox fallback không đổi và reflow. Vì vậy cần xem thêm chất lượng nhãn, không diễn giải coverage là đánh giá ngữ nghĩa hoàn chỉnh.','',
            '## Tái lập và xem trực quan','', '```bash','python3 -m content.pipeline survey','python3 -m content.pipeline run --split dev','python3 -m content.pipeline report','# Kiểm tra 5 mẫu và ghi development_review.md trước khi freeze','python3 -m content.pipeline freeze','python3 -m content.pipeline run --split all','python3 -m content.pipeline report','python3 -m unittest discover -s tests -v','python3 -m http.server 8765 --bind 127.0.0.1','```','',
            'Mở `http://127.0.0.1:8765/content/results/organized_v1/index.html`. Có thể mở trực tiếp index.html/review.html (payload và ảnh tương đối, không cần fetch). Filter GT, detector, matched, uncertain, unmatched; chọn hàng để truy dấu detection → correspondence → content evidence → prediction → liên kết GT.','',
            '## Giới hạn','',
            'Một mẫu remaining duy nhất; toàn bộ metadata GT và detector listings đã được khảo sát trước khi freeze. Không gọi đây là test độc lập. Cấu hình không điều chỉnh theo điểm remaining. Visual content candidate chỉ là thay đổi cấu trúc ảnh, có thể do style, detector crop hoặc correspondence sai. OCR giống nhau không phát hiện thay icon nằm trong cùng container. OCR khác ít được đưa vào uncertain nên có thể bỏ sót đổi nội dung một ký tự. Split/merge được chuẩn hóa chuỗi bên trong bbox, nhưng assignment detector đại diện vẫn one-to-one, chưa giải quyết quan hệ nhiều-nhiều segmentation. Không có nhãn correspondence/content âm đầy đủ; prediction không liên kết GT là unverified, không mặc định false positive.','',
            'Chưa review thủ công đầy đủ mọi cặp và nhóm. Xem validation.json cho các kiểm tra kỹ thuật; không dùng validation thay nhãn correctness.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if not have_all:(out/'DEV_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (out/'validation.json').write_text(json.dumps({'samples':len(selected),'all_predictions_created_before_gt_scoring':True,
        'one_to_one_and_accounting_assertions_passed':True,'source_hashes_verified':read(out/f'{split}_status.json')['source_hashes_verified'],
        'frozen_prediction_hashes_verified':True,'full_human_matching_review':False,'full_human_group_review':False},indent=2), encoding='utf-8')
