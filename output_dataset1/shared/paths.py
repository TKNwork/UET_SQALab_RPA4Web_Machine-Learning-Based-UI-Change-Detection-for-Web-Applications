"""Resolve historical provenance paths after the documented folder migration."""
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parent.parent


def resource_path(name, default):
    """Environment overrides are absolute or relative to the repository root."""
    return (ROOT.parent / os.environ.get('RPA_' + name.upper(), str(default))).resolve()


def groundtruth_root():
    # Do not silently substitute the paired/native GT archive: its protocol differs.
    return resource_path('groundtruth', ROOT / 'groundtruth')
RELOCATIONS = {
    'geometry_pairs': 'geometry/pairs',
    'geometry_results': 'geometry/results',
    'geometry_study_first5': 'geometry/study_first5',
    'report_viewer': 'geometry/report_viewer',
    'content_results': 'content/results',
    'new_pipeline.py': 'docs/migrations/2026-10-21/code_before_split/new_pipeline.py',
    'prediction_groups.py': 'docs/migrations/2026-10-21/code_before_split/prediction_groups.py',
    'content_pipeline.py': 'docs/migrations/2026-10-21/code_before_split/content_pipeline.py',
}

def resolve_recorded_path(value):
    """Historical manifests stay immutable; resolve only the known moved prefix."""
    path = Path(value)
    if path.exists():return path
    # Parse Windows provenance on Linux and Linux provenance on Windows.
    recorded = PureWindowsPath(str(value)) if '\\' in str(value) else PurePosixPath(str(value))
    parts = recorded.parts
    if 'output_dataset1' in parts:
        relative = Path(*parts[parts.index('output_dataset1') + 1:])
    elif 'dataset1' in parts:
        return resource_path('dataset1', ROOT.parent / 'dataset1') / Path(*parts[parts.index('dataset1') + 1:])
    else:
        if recorded.is_absolute(): return path
        relative = Path(*parts)
    if not relative.parts:return path
    prefix = RELOCATIONS.get(relative.parts[0])
    return ROOT/prefix/Path(*relative.parts[1:]) if prefix else ROOT/relative
