"""Match residual OCR boxes after Geometry and score their individual envelopes."""
import argparse
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
from scipy.optimize import linear_sum_assignment

from content import pipeline as content
from content.scripts.evaluate_first500 import digest, normalize_gt
from content.text_changes import normalize_text
from geometry import pipeline as geometry
from shared.matching import load_normalized
from shared.regions import area, enclosure, intersect, iou

ROOT = Path(__file__).resolve().parents[2]
IOU_THRESHOLD = .30
CCR_THRESHOLD = .50


def passes(iou_value, ccr):
    return iou_value + 1e-12 >= IOU_THRESHOLD or ccr + 1e-12 >= CCR_THRESHOLD


def residual_text(pair_dir, result, side):
    detections, warnings = load_normalized(pair_dir/f'{side}.json')
    text = [e for e in detections if e['source'] == 'ocr']
    removed = {member for pair in result['matched_pairs']
               if pair['before_source'] == pair['after_source'] == 'ocr'
               for member in (pair[f'{side}_id'], *pair[f'{side}_group_members'])}
    return {'text_count': len(text), 'removed_text_ids': [e['id'] for e in text if e['id'] in removed],
            'remaining': [e for e in text if e['id'] not in removed], 'warnings': warnings}


def match_residual(before, after):
    # A zero assignment represents rejection. Keep eligible edges only.
    scores = np.zeros((len(before), len(after)))
    eligible = {}
    for i, baseline in enumerate(before):
        for j, inject in enumerate(after):
            b, a = baseline['bbox'], inject['bbox']
            intersection = area(intersect(b, a))
            similarity, ccr = iou(b, a), intersection/area(b)
            if passes(similarity, ccr):
                scores[i, j] = similarity + .10*ccr
                eligible[i, j] = (similarity, ccr, intersection/area(a))
    pairs = []
    for i, j in zip(*linear_sum_assignment(scores, maximize=True)):
        if (i, j) not in eligible:
            continue
        baseline, inject = before[i], after[j]
        b, a = baseline['bbox'], inject['bbox']
        bt, at = (normalize_text(element['class_label']) for element in (baseline, inject))
        similarity, ccr_baseline, ccr_inject = eligible[i, j]
        pairs.append({'pair_id': f'C{len(pairs)+1}', 'baseline_id': baseline['id'], 'inject_id': inject['id'],
                      'baseline_text': bt, 'inject_text': at, 'baseline_bbox': b, 'inject_bbox': a,
                      'different_text': bool(bt and at and bt != at),
                      'iou': similarity, 'ccr_baseline': ccr_baseline, 'ccr_inject': ccr_inject,
                      'envelope': enclosure([b, a])})
    return pairs


def score_gt(targets, pairs):
    rows = []
    for target in targets:
        links = []
        for pair in pairs:
            metrics = content.support_metrics([pair['envelope']], target['xyxy'])
            if passes(metrics['iou'], metrics['gt_fraction']):
                union = content.support_metrics([pair['baseline_bbox'], pair['inject_bbox']], target['xyxy'])
                links.append({'prediction_id': pair['pair_id'], 'envelope': pair['envelope'],
                              'iou': metrics['iou'], 'ccr_gt': metrics['gt_fraction'],
                              'different_text': pair['different_text'],
                              'actual_boxes_also_pass': passes(union['iou'], union['gt_fraction'])})
        rows.append({**target, 'pass': bool(links), 'links': links})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--count', type=int, default=50)
    args = parser.parse_args()
    out = args.output_root.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'content/results') or out == ROOT/'content/results':
        parser.error('Choose a new child directory under content/results')
    inventory = content.read(args.input_manifest)
    by_id = {s['pair_id']: s for s in inventory['samples']}
    ids = [f'{i:03d}' for i in range(args.count)]
    if not ids or any(pid not in by_id or by_id[pid]['errors'] for pid in ids):
        parser.error('Input manifest must cover every requested ID')
    out.mkdir(parents=True)
    content.write(out/'run_status.json', {'status': 'matching_without_gt'})
    frozen = []
    for pid in ids:
        sources = by_id[pid]['sources']
        pair_dir = out/'inputs'/pid
        pair_dir.mkdir(parents=True)
        for side, source in [('before', 'baseline'), ('after', 'inject')]:
            copyfile(sources[source], pair_dir/f'{side}.json')
        cfg = {**geometry.DEFAULT_CONFIG, 'image_root': str(Path(sources['dataset1']).parent)}
        result = geometry.process_pair(pair_dir, out, cfg)
        sides = {side: residual_text(pair_dir, result, side) for side in ('before', 'after')}
        pairs = match_residual(sides['before']['remaining'], sides['after']['remaining'])
        prediction_file = out/pid/'residual_predictions.json'
        content.write(prediction_file, {'pair_id': pid, 'sides': sides, 'predictions': pairs})
        frozen.append({'pair_id': pid, 'sha256': digest(prediction_file),
                       'baseline_remaining': len(sides['before']['remaining']),
                       'inject_remaining': len(sides['after']['remaining']),
                       'predictions': len(pairs), 'different_text': sum(p['different_text'] for p in pairs)})
        print(f'{pid}: {len(pairs)} residual pairs', flush=True)
    content.write(out/'prediction_manifest.json', {'phase': 'frozen_before_gt',
                  'matching_rule': 'IoU >= .30 OR intersection / baseline bbox area >= .50; one-to-one maximize IoU + .10 * CCR',
                  'samples': frozen})
    content.write(out/'run_status.json', {'status': 'scoring_gt'})
    rows, samples = [], []
    for artifact in frozen:
        pid = artifact['pair_id']
        pred_path = out/pid/'residual_predictions.json'
        if digest(pred_path) != artifact['sha256']:
            raise ValueError(f'{pid}: frozen predictions changed')
        gt_path = Path(by_id[pid]['sources']['groundtruth'])
        evaluated = [{**r, 'pair_id': pid} for r in score_gt(
            normalize_gt(content.read(gt_path)), content.read(pred_path)['predictions'])]
        rows.extend(evaluated)
        content.write(out/pid/'gt_join.json', evaluated)
        samples.append({'pair_id': pid, 'gt': len(evaluated), 'passed': sum(r['pass'] for r in evaluated),
                        'full_pass': bool(evaluated) and all(r['pass'] for r in evaluated),
                        'gt_sha256': digest(gt_path)})
    covered = sum(r['pass'] for r in rows)
    summary = {'samples': len(ids), 'content_positive_samples': sum(s['gt'] > 0 for s in samples),
               'content_gt': len(rows), 'passed_gt': covered, 'missed_gt': len(rows)-covered,
               'coverage': covered/len(rows) if rows else None,
               'full_pass_samples': sum(s['full_pass'] for s in samples),
               'predictions': sum(r['predictions'] for r in frozen),
               'different_text_predictions': sum(r['different_text'] for r in frozen),
               'gt_with_different_text_link': sum(any(l['different_text'] for l in r['links']) for r in rows),
               'envelope_only_pass_gt': sum(r['pass'] and not any(l['actual_boxes_also_pass'] for l in r['links']) for r in rows),
               'gt_rule': 'Envelope IoU >= .30 OR intersection / GT area >= .50'}
    content.write(out/'summary.json', summary)
    content.write(out/'samples.json', samples)
    content.write(out/'gt_join.json', rows)
    content.write(out/'run_status.json', {'status': 'complete'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
