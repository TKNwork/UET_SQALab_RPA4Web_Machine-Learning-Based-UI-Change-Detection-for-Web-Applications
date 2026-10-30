"""Create a small private transfer pack or verify a restored pack. Never include keys."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--output', type=Path)
    action.add_argument('--verify', action='store_true')
    parser.add_argument('--include-model', action='store_true')
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads((ROOT / 'artifacts/asset_manifest.json').read_text(encoding='utf-8'))
        bad = []
        for item in manifest['files']:
            path = (ROOT / item['path']).resolve()
            if not path.is_relative_to(ROOT): raise ValueError('Manifest path escapes repository')
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                bad.append(item['path'])
        print(json.dumps({'checked': len(manifest['files']), 'missing_or_changed': bad}, indent=2))
        return int(bool(bad))
    selected = set()
    pairs = ROOT / 'output_dataset1/geometry/pairs'
    # Preserve original mixed GT cohort exactly for content preparation.
    selected.update(pairs.glob('*/groundtruth.json'))
    ids = ['000', '187', '418', '437', '451', '489', '492']
    for pid in ids:
        for side in ['before', 'after']:
            selected.add(pairs / pid / (side + '.json'))
        for side in ['baseline', 'inject']:
            detection = list((ROOT / 'output_dataset1' / side / 'json').glob(pid + '_*_full_hierarchy.json'))
            image_dirs = [p for p in (ROOT / 'dataset1').glob(pid + '_*') if p.is_dir()]
            if len(detection) != 1 or len(image_dirs) != 1:
                parser.error(f'Missing or ambiguous resources for {pid}/{side}')
            selected.add(detection[0]); selected.add(image_dirs[0] / (side + '.png'))
    if args.include_model: selected.add(ROOT / 'OCR + Yolo26/yolo/models/yl26best.pt')
    if len(list(pairs.glob('*/groundtruth.json'))) != 131:
        parser.error('Expected 131 original pair GT files; restore the original resource archive.')
    records = []
    for path in sorted(selected):
        if not path.is_file(): parser.error(f'Missing resource: {path.relative_to(ROOT)}')
        records.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {'scope': '7 sample images/detections + 131 original mixed GT; no DOM/full dataset/results',
                'ids': ids, 'files': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for item in records: bundle.write(ROOT / item['path'], item['path'])
        bundle.writestr('artifacts/asset_manifest.json', json.dumps(manifest, indent=2))
    print(json.dumps({'output': str(args.output), 'files': len(records), 'bytes': args.output.stat().st_size}, indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
