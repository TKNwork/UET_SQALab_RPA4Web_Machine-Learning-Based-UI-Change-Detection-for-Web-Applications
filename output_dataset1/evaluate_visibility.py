"""Run the unmatched visibility baseline and score it against flat GT."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from compare_changes import compare
from content.scripts.evaluate_first500 import normalize_gt
from content.pipeline import support_metrics
from shared.regions import area, enclosure


def with_envelopes(candidates, max_gap=24, max_expansion=2):
    """Add compact, same-side envelopes of neighboring candidates."""
    envelopes = []
    for i, left in enumerate(candidates):
        a = left['bbox']
        for j in range(i + 1, len(candidates)):
            right = candidates[j]
            if left['side'] != right['side']:
                continue
            b = right['bbox']
            gap_x = max(0, a[0] - b[2], b[0] - a[2])
            gap_y = max(0, a[1] - b[3], b[1] - a[3])
            if gap_x + gap_y > max_gap or (gap_x and gap_y):
                continue
            box = enclosure([a, b])
            if area(box) <= max_expansion * (area(a) + area(b)):
                envelopes.append({**left, 'bbox': box, 'members': [i, j],
                                  'evidence': 'same_side_neighbor_envelope'})
    return candidates + envelopes


def score(candidates, targets, *, iou=.30, ccr=.50):
    links = []
    weights = np.zeros((len(candidates), len(targets)))
    for i, candidate in enumerate(candidates):
        for j, target in enumerate(targets):
            metrics = support_metrics([candidate['bbox']], target['xyxy'])
            metrics['passes'] = metrics['iou'] + 1e-12 >= iou or metrics['gt_fraction'] + 1e-12 >= ccr
            if metrics['passes']:
                links.append({'candidate': i, 'target': j, **metrics})
                # Prioritize match count, then IoU among equal-cardinality assignments.
                weights[i, j] = min(weights.shape) + 1 + metrics['iou']
    rows, cols = linear_sum_assignment(weights, maximize=True)
    return {
        'covered': sorted({link['target'] for link in links}),
        'matches': [[int(i), int(j)] for i, j in zip(rows, cols) if weights[i, j] > 0],
        'links': links,
    }


def infer(job):
    before, after_root, images = job
    sample = before.stem.removesuffix('_full_hierarchy')
    image_dir = images / sample
    start = perf_counter()
    result = compare(before, after_root / before.name, image_dir / 'baseline.png',
                     image_dir / 'inject.png', reuse_geometry=True)
    return {'sample': sample, 'candidates': result['visibility_candidates'],
            'seconds': perf_counter() - start}


def main():
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before-root', 'after-root', 'images', 'gt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--expected-count', type=int, default=500)
    args = parser.parse_args()
    paths = sorted(args.before_root.glob('*_full_hierarchy.json'))
    if len(paths) != args.expected_count:
        parser.error(f'Expected {args.expected_count} before files, found {len(paths)}')
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / 'predictions.jsonl'
    jobs = [(path, args.after_root, args.images) for path in paths]
    with ProcessPoolExecutor(max_workers=args.workers) as pool, frozen.open('w', encoding='utf-8') as file:
        for i, row in enumerate(pool.map(infer, jobs), 1):
            file.write(json.dumps(row) + '\n')
            if i % 50 == 0:
                print(f'{i}/{len(jobs)} inferred', flush=True)

    with frozen.open(encoding='utf-8') as file, (args.output / 'scores.jsonl').open('w', encoding='utf-8') as scores:
        for line in file:
            row = json.loads(line)
            gt_path = args.gt / (row['sample'].split('_', 1)[0] + '.json')
            targets = normalize_gt(json.loads(gt_path.read_bytes()), 'visibility')
            scored = score(row['candidates'], targets)
            scores.write(json.dumps({**row, 'targets': targets, **scored}) + '\n')


if __name__ == '__main__':
    main()
