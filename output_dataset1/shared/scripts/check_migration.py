"""Verify moved data/result bytes and historical protected sources, read-only."""
import json
import hashlib
from pathlib import Path
from shared.paths import ROOT,resolve_recorded_path


def main():
    audit=ROOT/'docs/migrations/2026-10-21';manifest=json.loads((audit/'file_moves.json').read_text(encoding='utf-8'))
    checked=0;bad=[]
    for entry in manifest['files']:
        target=ROOT/entry['after']
        if not any(entry['after'].startswith(prefix) for prefix in ['geometry/pairs/','geometry/results/','geometry/study_first5/','content/results/','geometry/report_viewer/public/']):continue
        checked+=1
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=entry['sha256']:bad.append(entry['after'])
    protected=json.loads((ROOT/'docs/protected_files_2026-10-18.json').read_text(encoding='utf-8'));protected_bad=[]
    for entry in protected:
        target=resolve_recorded_path(entry['path'])
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=entry['sha256']:protected_bad.append(entry['path'])
    result={'moved_data_and_results_checked':checked,'moved_data_or_results_changed':bad,'historical_protected_files_checked':len(protected),'historical_protected_files_changed':protected_bad}
    (audit/'validation.json').write_text(json.dumps(result,indent=2)+'\n', encoding='utf-8');print(json.dumps(result,indent=2))
    assert not bad and not protected_bad

if __name__=='__main__':main()
