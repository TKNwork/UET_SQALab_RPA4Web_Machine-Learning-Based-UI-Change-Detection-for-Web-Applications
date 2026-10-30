"""Copy the original mixed GT from geometry pairs without changing its schema/cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, default=ROOT / 'output_dataset1/geometry/pairs')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/content_groundtruth')
    args = parser.parse_args()
    sources = sorted(args.pairs.glob('*/groundtruth.json'))
    if not sources: parser.error('No original pair GT files found; restore geometry pairs first.')
    cohort = {}
    for source in sources:
        data = json.loads(source.read_text(encoding='utf-8'))
        if 'bboxes' not in data: parser.error(f'Unexpected schema: {source}')
        count = sum('content' in b.get('type', []) for b in data['bboxes'])
        if count: cohort[source.parent.name] = count
    expected = {'187': 2, '418': 1, '437': 1, '451': 7, '489': 1, '492': 2}
    if cohort != expected: parser.error('Original 6-sample / 14-GT content cohort differs; review before proceeding.')
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sources:
        dest = args.output / (source.parent.name + '.json')
        if dest.exists() and dest.read_bytes() != source.read_bytes():
            raise FileExistsError(f'Refusing to replace different GT: {dest}')
        shutil.copy2(source, dest)
        records.append({'pair_id': source.parent.name, 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()})
    print(json.dumps({'copied': len(records), 'content_cohort': cohort, 'output': str(args.output)}, indent=2))


if __name__ == '__main__': main()
