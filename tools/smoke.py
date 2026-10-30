"""Offline synthetic geometry + content smoke, no dataset, model or API needed."""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'output_dataset1'))


def main():
    from PIL import Image
    from geometry import pipeline as geometry
    from content import pipeline as content
    with tempfile.TemporaryDirectory(prefix='rpa-smoke-') as folder:
        pair = Path(folder) / '000'; pair.mkdir()
        for side in ['before', 'after']:
            data = {'elements': [{'id': side, 'source': 'ocr', 'class_label': 'Submit',
                                  'bbox': [10, 10, 100, 30], 'confidence': .99}]}
            (pair / f'{side}.json').write_text(json.dumps(data), encoding='utf-8')
            Image.new('RGB', (160, 80), 'white').save(pair / f'{side}.png')
        result = geometry.process_pair(pair, Path(folder) / 'out', geometry.DEFAULT_CONFIG)
        assert len(result['matched_pairs']) == 1, 'Geometry correspondence failed'
        result = content.infer(pair / 'before.json', pair / 'after.json',
                               pair / 'before.png', pair / 'after.png', content.CONFIG)
        assert len(result['matched_pairs']) == 1, 'Content correspondence failed'
    print('PASS: synthetic geometry + content, no API calls and no research files changed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
