"""Audit or export files selected by .gitignore, without changing the local Git state."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.agents', '.aws', '.codex'}
TOKEN = re.compile(rb'AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def candidates():
    paths = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for name in files:
            path = Path(base) / name
            paths.append(path.relative_to(ROOT).as_posix())
    # Temporary Git metadata lets Git interpret nested .gitignore rules exactly.
    with tempfile.TemporaryDirectory(prefix='rpa-git-audit-') as folder:
        subprocess.run(['git', 'init', '--bare', '-q', folder], check=True)
        result = subprocess.run(['git', '-c', 'core.excludesFile=' + os.devnull,
                                 '--git-dir=' + folder, '--work-tree=' + str(ROOT),
                                 'check-ignore', '--no-index', '-z', '--stdin'],
                                input=('\0'.join(paths) + '\0').encode('utf-8'), capture_output=True)
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode('utf-8', errors='replace'))
        ignored = set(result.stdout.decode('utf-8').rstrip('\0').split('\0'))
    return sorted(set(paths) - ignored)


def audit(paths):
    records, issues, folded = [], [], {}
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{p}{n}' for p in ['COM', 'LPT'] for n in range(1, 10)}
    for name in paths:
        path = ROOT / name
        if path.is_symlink():
            issues.append({'path': name, 'reason': 'symlink requires review for Windows'})
            continue
        data = path.read_bytes()
        if len(data) > 10 * 1024**2: issues.append({'path': name, 'reason': 'source file exceeds 10 MiB'})
        if TOKEN.search(data): issues.append({'path': name, 'reason': 'possible secret; value omitted'})
        for part in PureWindowsPath(name).parts:
            if any(c in part for c in '<>:"|?*') or part.endswith((' ', '.')) or part.split('.')[0].upper() in reserved:
                issues.append({'path': name, 'reason': 'Windows-incompatible filename'})
        if len(name) > 180: issues.append({'path': name, 'reason': 'long relative path; review Windows checkout root'})
        key = name.casefold()
        if key in folded: issues.append({'path': name, 'reason': 'case-insensitive collision with ' + folded[key]})
        folded[key] = name
        records.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    return {'files': len(records), 'bytes': sum(r['bytes'] for r in records), 'issues': issues, 'inventory': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Create a new source zip; never overwrite an existing file')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = audit(candidates())
    print(json.dumps({k: v for k, v in report.items() if k != 'inventory'}, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if report['issues']: return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
            for record in report['inventory']:
                path = ROOT / record['path']
                if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                    raise RuntimeError(f'File changed during export: {record["path"]}')
                bundle.write(path, record['path'])
        print('Source bundle:', args.output)
    return 0


if __name__ == '__main__': raise SystemExit(main())
