#!/usr/bin/env python3
"""Portable launcher. Relative paths are always relative to this repository."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MATCHING = ROOT / 'output_dataset1'


def execute(args, cwd=MATCHING, env=None):
    return subprocess.call([sys.executable, '-X', 'utf8', *map(str, args)], cwd=cwd,
                           env=env or os.environ.copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.local.json')
    parser.add_argument('flow', choices=['doctor', 'smoke', 'test', 'geometry', 'content', 'perception'])
    parser.add_argument('extra', nargs=argparse.REMAINDER, help='Arguments passed to the selected flow')
    args = parser.parse_args()
    extra = args.extra
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    defaults = json.loads((ROOT / 'config.example.json').read_text(encoding='utf-8'))
    defaults.update(config)
    paths = {key: (ROOT / value).resolve() for key, value in defaults.items()}
    for key in paths:
        if key not in config and os.environ.get('RPA_' + key.upper()):
            paths[key] = (ROOT / os.environ['RPA_' + key.upper()]).resolve()
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    for key in ['dataset1', 'baseline', 'inject', 'groundtruth', 'yolo_model']:
        env['RPA_' + key.upper()] = str(paths[key])
    if args.flow in ['doctor', 'smoke']:
        return execute([ROOT / 'tools' / (args.flow + '.py'), *extra], ROOT, env)
    if args.flow == 'test':
        if extra:
            parser.error('test does not accept extra arguments')
        return execute(['-m', 'unittest', 'discover', '-s', '.', '-p', 'test_*.py', '-v'], env=env)
    # User-supplied relative file paths refer to the repository, not child cwd.
    path_options = {'--input-root', '--output-root', '--image-root', '--image', '--output'}
    for i, item in enumerate(extra):
        if '=' in item and item.split('=', 1)[0] in path_options:
            flag, value = item.split('=', 1)
            extra[i] = flag + '=' + str((ROOT / value).resolve())
        elif i and extra[i-1] in path_options:
            extra[i] = str((ROOT / item).resolve())
    if args.flow == 'geometry':
        if '--help' not in extra and '-h' not in extra:
            if not any(x == '--input-root' or x.startswith('--input-root=') for x in extra):
                extra += ['--input-root', str(paths['geometry_pairs'])]
            if not any(x == '--image-root' or x.startswith('--image-root=') for x in extra):
                extra += ['--image-root', str(paths['dataset1'])]
        return execute(['-m', 'geometry.match_geometry', *extra], env=env)
    if args.flow == 'content':
        return execute(['-m', 'content.pipeline', *extra], env=env)
    return execute([ROOT / 'OCR + Yolo26/run_perception.py', *extra], ROOT, env)


if __name__ == '__main__':
    raise SystemExit(main())
