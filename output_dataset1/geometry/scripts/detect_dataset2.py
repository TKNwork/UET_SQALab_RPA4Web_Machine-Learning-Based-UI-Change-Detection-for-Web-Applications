"""Run unchanged dataset1 perception functions on dataset2, with strict OCR caching."""
import ast
import hashlib
import gzip
import json
import logging
import sys
import time
import types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
UP=ROOT.parent/'OCR + Yolo26'
OUT=ROOT/'dataset2_detection'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2), encoding='utf-8');tmp.replace(p)

def load_upstream():
    sys.path.insert(0,str(UP))
    path=UP/'run_dataset1_combine_pipeline.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    # Suppress only the module-level output mkdir. Inference/fusion function bodies unchanged.
    tree.body=[n for n in tree.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='os.makedirs')]
    mod=types.ModuleType('dataset2_upstream');mod.__file__=str(path)
    exec(compile(tree,str(path),'exec'),mod.__dict__)
    return mod

def main():
    OUT.mkdir(exist_ok=True)
    m=load_upstream()
    from ultralytics import YOLO
    model_path=UP/'yolo/models/yl26best.pt'
    device=m.detect_device();model=YOLO(str(model_path));key=m.load_api_key(m.KEY_FILE)
    import torch
    torch.set_num_threads(min(4,torch.get_num_threads()))
    original_post=m.requests.post
    class OCRFailure(BaseException):pass
    current={}
    def post(url,**kwargs):
        cp=current['raw']
        if cp.exists():
            with gzip.open(cp,'rt',encoding='utf-8') as f:data=json.load(f)
        else:
            data=None
            for attempt in range(3):
                try:
                    response=original_post(url,**kwargs)
                    body=response.json()
                    if response.status_code==200 and body.get('responses') and not body['responses'][0].get('error'):
                        data=body;cp.parent.mkdir(parents=True,exist_ok=True)
                        with gzip.open(cp,'wt',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False)
                        break
                    code=body.get('error',body.get('responses',[{}])[0].get('error',{})).get('code',response.status_code)
                    if code in (400,401,403):raise OCRFailure(f'OCR rejected request code {code}')
                except OCRFailure:raise
                except Exception:pass
                time.sleep(2*(attempt+1))
            if data is None:raise OCRFailure('OCR failed after 3 attempts; no key or request URL logged')
        class Response:
            status_code=200
            def json(self):return data
        return Response()
    m.requests.post=post
    # Do not let upstream exception logger expose requests URLs/credentials.
    m.logger.setLevel(logging.CRITICAL)
    # Display is handled by the geometry viewer; omit redundant detector PNGs.
    m.visualize_and_save=lambda *args,**kwargs:None
    entries=[]
    for folder in sorted((ROOT/'dataset2').iterdir(),key=lambda p:int(p.name.split('_')[0])):
        if not folder.is_dir():continue
        for side,name in [('baseline','baseline.png'),('inject','inject.png')]:
            image=folder/name
            entries.append((folder,side,image))
    code={str(p.relative_to(UP)):sha(p) for p in UP.rglob('*.py') if not any(x in p.parts for x in ('__pycache__','.venv','venv'))}
    config={'upstream':str(UP),'entrypoint':'run_dataset1_combine_pipeline.process_image','model_sha256':sha(model_path),'code_hashes':code,'device':device,'yolo_thresholds':m.SIZE_THRESHOLDS,'ocr_confidence':.8,'cv_min_rect_conf':.9,'cv_min_divider_conf':.85,'wbf_iou':.85,'historical_source_version_proven':False,'note':'Reuse current dataset1 entrypoint unchanged; historical upstream source hash unavailable.'}
    old=OUT/'config.json'
    if old.exists():
        prior=json.loads(old.read_text(encoding='utf-8'))
        if prior['model_sha256']!=config['model_sha256'] or prior['code_hashes']!=code:raise RuntimeError('Detector code/model changed; use a separate output run')
    save(old,config)
    statuses=[];service_blocked=False
    for idx,(folder,side,image) in enumerate(entries):
        name=folder.name;dest=OUT/side/'json'/f'{name}_full_hierarchy.json';vis=OUT/side/'visualized'/f'{name}_full_hierarchy.png';receipt=OUT/side/'receipts'/f'{name}.json'
        raw=OUT/side/'ocr_raw'/f'{name}.json.gz';current['raw']=raw
        item={'pair_id':name.split('_')[0],'side':side,'image':str(image),'output':str(dest),'ocr_raw':str(raw)}
        try:
            image_hash=sha(image)
            if receipt.exists():
                r=json.loads(receipt.read_text(encoding='utf-8'))
                if r['image_sha256']!=image_hash or not dest.exists() or r['output_sha256']!=sha(dest):raise RuntimeError('Resume provenance mismatch')
                item.update(status='complete',resumed=True)
            else:
                if service_blocked:item.update(status='blocked_ocr_service');statuses.append(item);continue
                if dest.exists():
                    # Our interrupted new-run artifact is quarantined, never trusted as a cache.
                    quarantine=dest.with_suffix('.unverified.json')
                    if quarantine.exists():raise RuntimeError('Conflicting unverified output')
                    dest.rename(quarantine)
                m.process_image(str(image),name,model,device,key,str(dest),str(vis))
                result=json.loads(dest.read_text(encoding='utf-8'))
                dest.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')), encoding='utf-8')
                assert raw.exists() and result['elements'] and all(len(e['bbox'])==4 for e in result['elements'])
                r={'image_sha256':image_hash,'output_sha256':sha(dest),'ocr_raw_sha256':sha(raw),'elements':len(result['elements']),'sources':sorted({e['source'] for e in result['elements']}),'ocr_yolo_cv_executed':True}
                save(receipt,r);item.update(status='complete',elements=r['elements'])
        except OCRFailure as exc:
            item.update(status='failed_ocr',error=str(exc))
            if 'rejected' in str(exc):service_blocked=True
        except Exception as exc:
            item.update(status='failed',error=type(exc).__name__)
        statuses.append(item)
        save(OUT/'manifest.json',{'total_images':len(entries),'processed_images':len(statuses),'images':statuses})
        save(OUT/'run_status.json',{'status':'running','processed':len(statuses),'total':len(entries),'complete':sum(s['status']=='complete' for s in statuses),'service_blocked':service_blocked})
        print(f"[{idx+1}/{len(entries)}] {name}/{side}: {item['status']}",flush=True)
    save(OUT/'manifest.json',{'total_images':len(entries),'processed_images':len(statuses),'images':statuses})
    save(OUT/'run_status.json',{'status':'complete' if all(s['status']=='complete' for s in statuses) else 'partial_or_blocked','processed':len(statuses),'total':len(entries),'complete':sum(s['status']=='complete' for s in statuses),'service_blocked':service_blocked})
if __name__=='__main__':main()
