"""Check local prerequisites without sending requests or loading model weights."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flow', choices=['matching', 'perception', 'gemini'], default='matching')
    parser.add_argument('--require-assets', action='store_true')
    args = parser.parse_args()
    modules = ['numpy', 'scipy', 'PIL']
    if args.flow == 'perception': modules += ['requests', 'cv2', 'torch', 'ultralytics']
    if args.flow == 'gemini': modules += ['requests', 'google.genai']
    checks = {}
    for name in modules:
        try: checks[name] = importlib.util.find_spec(name) is not None
        except ModuleNotFoundError: checks[name] = False
    def configured(name, default):
        return Path(os.environ.get('RPA_' + name.upper(), str(ROOT / default)))
    paths = {'dataset1': configured('dataset1', 'dataset1'),
             'baseline': configured('baseline', 'output_dataset1/baseline/json'),
             'inject': configured('inject', 'output_dataset1/inject/json'),
             'groundtruth': configured('groundtruth', 'output_dataset1/groundtruth'),
             'yolo_model': configured('yolo_model', 'OCR + Yolo26/yolo/models/yl26best.pt')}
    assets = {key: path.is_file() if key == 'yolo_model' else path.is_dir() and any(path.iterdir())
              for key, path in paths.items()}
    assets['groundtruth'] = any(paths['groundtruth'].glob('*.json'))
    # Report presence only; never print keys or proxy URLs.
    secrets = {name: bool(os.environ.get(name)) for name in ['GOOGLE_VISION_API_KEY', 'GEMINI_API_KEY']}
    print(json.dumps({'python': platform.python_version(), 'platform': platform.system(),
                      'modules': checks, 'assets_present': assets, 'environment_keys_present': secrets}, indent=2))
    required = ['dataset1', 'baseline', 'inject', 'groundtruth'] if args.flow == 'matching' else ['yolo_model'] if args.flow == 'perception' else []
    return int(sys.version_info < (3, 10) or not all(checks.values()) or
               (args.require_assets and not all(assets[k] for k in required)))


if __name__ == '__main__':
    raise SystemExit(main())
