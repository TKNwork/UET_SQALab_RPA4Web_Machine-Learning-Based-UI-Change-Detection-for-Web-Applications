"""Match OCR elements, freeze predictions, and score content GT with Geometry overlap."""
import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from content import pipeline as c
from content.hierarchy_text import load_hierarchy_text
from content.replacement_changes import raw_text
from content.scripts.evaluate_first500 import digest, normalize_gt
from shared.regions import area, enclosure

ROOT = Path(__file__).resolve().parents[2]
IOU, CCR = .30, .50


def infer_sample(job):
    pid, inputs, out, hierarchy_ocr = job
    result = c.infer(*inputs, ocr_only=True,
                     detection_loader=load_hierarchy_text if hierarchy_ocr else None)
    result = select_predictions(result, hierarchy_ocr)
    path = out/pid/'inference.json'
    c.write(path, result)
    return summarize(pid, path, result)


def select_predictions(result, hierarchy_ocr):
    predictions = []
    for pair in result['matched_pairs'] + result['uncertain_pairs']:
        before, _ = raw_text(pair, 'before', result['before_detections'])
        after, _ = raw_text(pair, 'after', result['after_detections'])
        evidence = pair['content']['visual']
        state = 'text_unavailable'
        if before and after:
            state = 'same_text' if before == after else 'different_text'
        pair['text_comparison'] = {
            'baseline_text': before, 'inject_text': after, 'state': state,
            'pixel_supported': evidence.get('changed_fraction', 0) >= c.CONFIG['ocr_min_visual_fraction'],
        }
        pair['envelope'] = enclosure([pair['before_bbox'], pair['after_bbox']])
    for pair in result['matched_pairs']:
        comparison = pair['text_comparison']
        if comparison['state'] == 'different_text' and comparison['pixel_supported']:
            predictions.append({**pair, 'prediction_index': len(predictions)})
    groups = c.groups(predictions, result['config'])
    for group in groups:
        group['envelope'] = enclosure(group['support_boxes'])
    profile = 'hierarchy-ocr-text-pairs-v1' if hierarchy_ocr else 'ocr-text-pairs-v1'
    result.update(predictions=predictions, groups=groups, profile=profile)
    return result


def summarize(pid, path, result):
    predictions, groups = result['predictions'], result['groups']
    return {'pair_id': pid, 'sha256': digest(path),
            'baseline_text': len(result['before_detections']),
            'inject_text': len(result['after_detections']),
            'baseline_ocr_on_nonocr_primary': sum(e.get('hierarchy_source', 'ocr') != 'ocr' for e in result['before_detections']),
            'inject_ocr_on_nonocr_primary': sum(e.get('hierarchy_source', 'ocr') != 'ocr' for e in result['after_detections']),
            'matched': len(result['matched_pairs']), 'uncertain': len(result['uncertain_pairs']),
            'unmatched_baseline': len(result['unmatched_before']),
            'unmatched_inject': len(result['unmatched_after']),
            'predictions': len(predictions), 'groups': len(groups)}


def overlap(boxes, gt):
    metrics = c.support_metrics(boxes, gt)
    return {**metrics, 'passes': metrics['iou']+1e-12 >= IOU or metrics['gt_fraction']+1e-12 >= CCR}


def score_sample(result, targets, *, accept_contained=False):
    rows = []
    for target in targets:
        box = target['xyxy']
        pair_links = []
        for pair in result['matched_pairs']:
            before = overlap([pair['before_bbox']], box)
            after = overlap([pair['after_bbox']], box)
            if before['passes'] or after['passes']:
                pair_links.append({'before_id': pair['before_id'], 'after_id': pair['after_id'],
                                   'baseline': before, 'inject': after,
                                   'comparison': pair['text_comparison']})
        links = []
        for kind, items in [('singleton', result['predictions']), ('group', result['groups'])]:
            for item in items:
                boxes = [item['before_bbox'], item['after_bbox']] if kind == 'singleton' else item['support_boxes']
                envelope = overlap([item['envelope']], box)
                x1, y1, x2, y2 = item['envelope']
                gx1, gy1, gx2, gy2 = box
                envelope['overlap_pass'] = envelope['passes']
                envelope['fully_contained'] = (area(item['envelope']) > 0 and
                    gx1 <= x1 and gy1 <= y1 and x2 <= gx2 and y2 <= gy2)
                envelope['passes'] |= accept_contained and envelope['fully_contained']
                union = overlap(boxes, box)
                if envelope['passes'] or union['passes']:
                    indices = [item['prediction_index']] if kind == 'singleton' else item['member_prediction_indices']
                    links.append({'kind': kind, 'prediction_indices': indices,
                                  'envelope': envelope, 'union': union,
                                  'envelope_gap_only': union['gt_fraction'] == 0})
        rows.append({**target, 'pair_links': pair_links, 'prediction_links': links,
                     'matched_pair_covered': bool(pair_links),
                     'both_endpoints_covered': any(l['baseline']['passes'] and l['inject']['passes'] for l in pair_links),
                     'different_ocr_pair_covered': any(l['comparison']['state'] == 'different_text' for l in pair_links),
                     'singleton_envelope_covered': any(l['kind'] == 'singleton' and l['envelope']['passes'] for l in links),
                     'singleton_envelope_overlap_covered': any(l['kind'] == 'singleton' and l['envelope']['overlap_pass'] for l in links),
                     'grouped_envelope_covered': any(l['envelope']['passes'] for l in links),
                     'grouped_envelope_overlap_covered': any(l['envelope']['overlap_pass'] for l in links),
                     'grouped_union_covered': any(l['union']['passes'] for l in links)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--full-hierarchy-ocr', action='store_true',
                        help='Read OCR provenance and text from fused full hierarchy JSON')
    parser.add_argument('--accept-contained-envelope', action='store_true',
                        help='Also pass an envelope that is fully inside the GT rectangle')
    args = parser.parse_args()
    out = args.output_root.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'content/results') or out == ROOT/'content/results':
        parser.error('Choose a new child directory under content/results')
    if args.workers < 1:
        parser.error('--workers must be positive')
    inventory = c.read(args.input_manifest)
    samples = inventory['samples']
    if not samples or any(s['errors'] for s in samples):
        parser.error('Input manifest must describe a complete cohort')
    out.mkdir(parents=True)
    c.write(out/'run_status.json', {'status': 'matching_without_gt'})
    jobs = [(s['pair_id'], [s['sources'][key] for key in
            ('baseline', 'inject', 'before_image', 'after_image')], out, args.full_hierarchy_ocr) for s in samples]
    source_hashes = {path: digest(Path(path)) for _, paths, _, _ in jobs for path in paths}
    frozen = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for item in pool.map(infer_sample, jobs):
            frozen.append(item)
            if len(frozen) % 25 == 0 or len(frozen) == len(jobs):
                print(f'Matched {len(frozen)}/{len(jobs)} samples', flush=True)
    if any(digest(Path(path)) != sha for path, sha in source_hashes.items()):
        raise ValueError('Detection or image source changed during matching')
    c.write(out/'prediction_manifest.json', {
        'phase': 'all_pairs_and_predictions_frozen_before_gt',
        'full_hierarchy_ocr': args.full_hierarchy_ocr,
        'matcher_config': c.CONFIG, 'normalization_config': c.shared.NORMALIZATION_CONFIG,
        'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                        [Path(__file__).resolve(), ROOT/'content/pipeline.py', ROOT/'shared/matching.py',
                         ROOT/'content/replacement_changes.py', ROOT/'content/text_changes.py', ROOT/'content/hierarchy_text.py',
                         ROOT/'shared/regions.py']},
        'input_sha256': source_hashes, 'samples': frozen})
    c.write(out/'run_status.json', {'status': 'scoring_content_gt'})
    rows = []
    gt_hashes = {}
    for sample, artifact in zip(samples, frozen):
        pid = sample['pair_id']
        path = out/pid/'inference.json'
        if digest(path) != artifact['sha256']:
            raise ValueError(f'{pid}: frozen prediction changed')
        gt_path = Path(sample['sources']['groundtruth'])
        gt_hashes[str(gt_path)] = digest(gt_path)
        targets = normalize_gt(c.read(gt_path))
        evaluated = [{**row, 'pair_id': pid} for row in
                     score_sample(c.read(path), targets, accept_contained=args.accept_contained_envelope)]
        rows.extend(evaluated)
        c.write(out/pid/'gt_join.json', evaluated)
    modes = ('matched_pair', 'both_endpoints', 'different_ocr_pair', 'singleton_envelope', 'grouped_envelope', 'grouped_union')
    counts = Counter()
    for record in frozen:
        counts.update({key: record[key] for key in ('matched', 'uncertain', 'unmatched_baseline',
                                                   'unmatched_inject', 'predictions', 'groups')})
    summary = {'samples': len(samples), 'content_gt': len(rows), 'matching_counts': dict(counts),
               'full_hierarchy_ocr': args.full_hierarchy_ocr,
               'input_text_counts': {key: sum(item[key] for item in frozen)
                   for key in ('baseline_text', 'inject_text', 'baseline_ocr_on_nonocr_primary', 'inject_ocr_on_nonocr_primary')},
               'coverage_rule': 'IoU >= 0.30 OR intersection / GT area >= 0.50',
               'accept_contained_envelope': args.accept_contained_envelope,
               'coverage': {key: {'covered': sum(row[key+'_covered'] for row in rows),
                                  'total': len(rows)} for key in modes},
               'envelope_gap_only_gt': sum(row['grouped_envelope_covered'] and
                   all(link['envelope_gap_only'] for link in row['prediction_links'] if link['envelope']['passes'])
                   for row in rows)}
    summary['envelope_comparison'] = {
        mode: {'overlap_only': sum(row[mode+'_envelope_overlap_covered'] for row in rows),
               'current_rule': summary['coverage'][mode+'_envelope']['covered'],
               'added_by_containment': sum(row[mode+'_envelope_covered'] and
                                           not row[mode+'_envelope_overlap_covered'] for row in rows)}
        for mode in ('singleton', 'grouped')}
    c.write(out/'summary.json', summary)
    c.write(out/'gt_join.json', rows)
    c.write(out/'missed_gt.json', [row for row in rows if not row['grouped_envelope_covered']])
    c.write(out/'gt_manifest.json', {'gt_sha256': gt_hashes})
    c.write(out/'run_status.json', {'status': 'complete'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
