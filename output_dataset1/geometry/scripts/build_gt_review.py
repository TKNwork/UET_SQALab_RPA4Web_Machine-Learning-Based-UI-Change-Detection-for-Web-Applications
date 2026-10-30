"""Create complete GT review, with inline thumbnails and local full-resolution images."""
import base64
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
VIS=ROOT/'geometry/results/gt_review'

def thumbnail(path):
    with Image.open(path) as source:
        size=list(source.size);im=source.convert('RGB');im.thumbnail((220,130))
        buf=BytesIO();im.save(buf,format='WEBP',quality=30,method=4)
    return 'data:image/webp;base64,'+base64.b64encode(buf.getvalue()).decode(),size


def main():
    VIS.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'geometry/results/matching_manifest.json').read_text(encoding='utf-8'))
    samples=[];full=[];hashes=[];types=Counter();count=fallback=0
    files=sorted((ROOT/'groundtruth').glob('*.json'))
    assert len(files)==len(manifest['pair_ids'])==131
    for p in files:
        pid=p.stem;raw=json.loads(p.read_text(encoding='utf-8'));hashes.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        assert raw==json.loads((ROOT/'geometry/pairs'/pid/'groundtruth.json').read_text(encoding='utf-8'))
        assert raw.get('pair_id')==pid
        bboxes=[]
        for idx,g in enumerate(raw['bboxes']):
            for key in ['x','y','w','h','selector','type','fallback']:assert key in g,(pid,key)
            assert g['w']>0 and g['h']>0
            bboxes.append(dict(g,index=idx));types.update(g['type']);count+=1;fallback+=bool(g['fallback'])
        result=json.loads((ROOT/'geometry/results'/pid/'matching_result_full.json').read_text(encoding='utf-8'))
        item={'id':pid,'bboxes':bboxes,'size':{}}
        for side,path in result['image_paths'].items():item[side],item['size'][side]=thumbnail(path)
        samples.append(item)
        expanded=dict(item,previewBefore=item['before'])
        for side in ['before','after']:
            public=ROOT/'geometry/report_viewer/public/data/geometry-v3'/pid/(side+'.png')
            assert public.is_file()
            expanded[side]='../data/geometry-v3/'+pid+'/'+side+'.png'
        full.append(expanded)
    totals={'total_boxes':count,'geometry_boxes':types['geometry'],'fallback_boxes':fallback}
    assert count==297 and types['geometry']==270
    template=(ROOT/'geometry/scripts/gt_review_fragment.html').read_text(encoding='utf-8')
    def render(samples,thumb):
        payload=json.dumps(dict(samples=samples,thumbnail=thumb,**totals),ensure_ascii=False,separators=(',',':')).replace('<',r'<')
        return template.replace('__GT_DATA__',payload)
    VIS.mkdir(parents=True,exist_ok=True)
    fragment=render(samples,True)
    assert len(fragment.encode())<1_000_000
    (VIS/'full-ground-truth.html').write_text(fragment, encoding='utf-8')
    target=ROOT/'geometry/report_viewer/public/gt';target.mkdir(exist_ok=True)
    styles='''<style>
    :root{color-scheme:light dark;--background:light-dark(#fff,#101827);--foreground:light-dark(#172033,#eef2f8);--muted:light-dark(#edf1f5,#223047);--viz-series-1:#e63f52;--viz-series-2:#d28718;--viz-series-3:#2c9bdf;--viz-series-4:#b066e0;font:14px system-ui}
    body{margin:0;padding:20px;background:var(--background);color:var(--foreground)}
    .btn,select{font:inherit;padding:8px 10px;border:1px solid light-dark(#bac3d0,#526078);border-radius:6px;background:var(--muted);color:var(--foreground)}
    .btn[aria-pressed=true]{outline:2px solid var(--viz-series-3)}
    .btn:disabled{opacity:.4}.cursor-interaction,.btn{cursor:pointer}.viz-controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:12px 0}.form-label,.form-check{display:flex;align-items:center;gap:6px}.text-muted{opacity:.8}
    h2{font-size:22px;font-weight:500}main{max-width:1500px;margin:auto}
    </style>'''
    document='<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Toàn bộ GT — 131 mẫu</title>'+styles+'<main>'+render(full,False)+'</main></html>'
    (target/'index.html').write_text(document, encoding='utf-8')
    stats={'samples':len(samples),'bboxes':count,'type_tags':dict(types),'fallback_bboxes':fallback,'inline_bytes':len(fragment.encode()),'all_source_gt_equal_geometry_pair_gt':True,'sources':hashes}
    (target/'validation.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2), encoding='utf-8')
    for entry in hashes:assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
    print(json.dumps({k:v for k,v in stats.items() if k!='sources'},ensure_ascii=False,indent=2))
    print('Inline:',VIS/'full-ground-truth.html');print('Full resolution:',target/'index.html')

if __name__=='__main__':main()
