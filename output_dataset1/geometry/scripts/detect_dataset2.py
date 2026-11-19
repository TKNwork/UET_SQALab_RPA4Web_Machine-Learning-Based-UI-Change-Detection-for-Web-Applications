"""Run the dataset1 detector in batches, with strict OCR caching."""
import argparse
import importlib.util
import os
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
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding='utf-8')
    # Windows readers and file sync can temporarily lock the destination.
    for attempt in range(6):
        try:
            tmp.replace(p)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * (attempt + 1))

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=ROOT/'dataset2')
    parser.add_argument('--output-root', type=Path, default=OUT)
    parser.add_argument('--model', type=Path, default=Path(os.environ.get('RPA_YOLO_MODEL') or UP/'yolo/models/yl26best.pt'))
    parser.add_argument('--side', choices=('baseline', 'inject', 'both'), default='both')
    parser.add_argument('--first-count', type=int, help='Require exactly the IDs 000 through N-1')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.first_count is not None and args.first_count < 1:
        parser.error('--first-count must be positive')
    sides = ('baseline', 'inject') if args.side == 'both' else (args.side,)
    folders = {}
    for folder in args.dataset_root.iterdir():
        prefix = folder.name.split('_')[0]
        if folder.is_dir() and prefix.isdigit():
            folders.setdefault(int(prefix), []).append(folder)
    ids = range(args.first_count) if args.first_count is not None else sorted(folders)
    entries = []
    for number in ids:
        matches = folders.get(number, [])
        if len(matches) != 1:
            parser.error(f'Expected one sample directory for ID {number:03d}, found {len(matches)}')
        folder = matches[0]
        for side in sides:
            image = folder/f'{side}.png'
            if not image.is_file():
                parser.error(f'Missing screenshot: {image}')
            entries.append((folder, side, image))
    if not entries:
        parser.error('No screenshots selected')
    missing = []
    if not args.model.is_file():
        missing.append(f'YOLO weights: {args.model}')
    if not os.environ.get('GOOGLE_VISION_API_KEY', '').strip() and not any(
        p.is_file() and p.read_text(encoding='utf-8-sig').strip()
        for p in (UP/'config/ocrkey.txt', UP/'ocrkey.txt')
    ):
        missing.append('Google Vision OCR key')
    for module in ('requests', 'cv2', 'torch', 'ultralytics'):
        if importlib.util.find_spec(module) is None:
            missing.append(f'Python module: {module}')
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.preflight or missing:
        report = {'status': 'blocked' if missing else 'ready', 'selected_images': len(entries),
                  'sides': list(sides), 'missing': missing}
        save(out/'preflight.json', report)
        print(json.dumps(report, indent=2))
        return 2 if missing else 0
    m=load_upstream()
    from ultralytics import YOLO
    model_path=args.model.resolve()
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
    code={str(p.relative_to(UP)):sha(p) for p in UP.rglob('*.py') if not any(x in p.parts for x in ('__pycache__','.venv','venv'))}
    config={'upstream':str(UP),'entrypoint':'run_dataset1_combine_pipeline.process_image','model_sha256':sha(model_path),'code_hashes':code,'device':device,'yolo_thresholds':m.SIZE_THRESHOLDS,'ocr_confidence':.8,'cv_min_rect_conf':.9,'cv_min_divider_conf':.85,'wbf_iou':.85,'historical_source_version_proven':False,'note':'Reuse current dataset1 entrypoint unchanged; historical upstream source hash unavailable.'}
    old=out/'config.json'
    if old.exists():
        prior=json.loads(old.read_text(encoding='utf-8'))
        if prior['model_sha256']!=config['model_sha256'] or prior['code_hashes']!=code:raise RuntimeError('Detector code/model changed; use a separate output run')
    save(old,config)
    statuses=[];service_blocked=False
    for idx,(folder,side,image) in enumerate(entries):
        name=folder.name;dest=out/side/'json'/f'{name}_full_hierarchy.json';vis=out/side/'visualized'/f'{name}_full_hierarchy.png';receipt=out/side/'receipts'/f'{name}.json'
        raw=out/side/'ocr_raw'/f'{name}.json.gz';current['raw']=raw
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
        save(out/'manifest.json',{'total_images':len(entries),'processed_images':len(statuses),'images':statuses})
        save(out/'run_status.json',{'status':'running','processed':len(statuses),'total':len(entries),'complete':sum(s['status']=='complete' for s in statuses),'service_blocked':service_blocked})
        print(f"[{idx+1}/{len(entries)}] {name}/{side}: {item['status']}",flush=True)
    save(out/'manifest.json',{'total_images':len(entries),'processed_images':len(statuses),'images':statuses})
    save(out/'run_status.json',{'status':'complete' if all(s['status']=='complete' for s in statuses) else 'partial_or_blocked','processed':len(statuses),'total':len(entries),'complete':sum(s['status']=='complete' for s in statuses),'service_blocked':service_blocked})
    return 0 if all(s['status']=='complete' for s in statuses) else 1


if __name__=='__main__':
    raise SystemExit(main())
