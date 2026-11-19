"""Bounded, two-phase content audit: inputs -> frozen inference -> GT join.

Run from output_dataset1: python -m content.scripts.evaluate_first500 --help
No detector rerun, metadata-derived GT, or geometry mutation.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from content import pipeline as c

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
CODE_ROOT = ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_index(root, pattern):
    result = {}
    for path in sorted(root.glob(pattern)):
        prefix = path.name.split('_')[0].split('.')[0]
        if prefix.isdigit():
            result.setdefault(int(prefix), []).append(path)
    return result


def unique(index, number):
    values = index.get(number, [])
    return values[0] if len(values) == 1 else None


def normalize_gt(data, change_type="content"):
    """Retain original indices and attributes; explicitly reject native GT."""
    if not isinstance(data.get('bboxes'), list):
        raise ValueError('Expected original flat bboxes schema; no native-GT fallback')
    targets = []
    for i, box in enumerate(data['bboxes']):
        if not isinstance(box.get('type'), list):
            raise ValueError(f'GT {i}: type must be a list')
        if change_type not in box['type']:
            continue
        values = [box[k] for k in ('x', 'y', 'w', 'h')]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError(f'GT {i}: non-finite coordinates')
        x, y, w, h = values
        if w <= 0 or h <= 0:
            raise ValueError(f'GT {i}: nonpositive size')
        targets.append({'original_gt_index': i, 'xyxy': [x, y, x+w, y+h],
                        'original': box, 'human_validity': None})
    return targets


def save_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-id', type=int, default=0,
                        help='Default cohort = 000..499, including missing IDs')
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--text-only', action='store_true',
                        help='Visible text changes only; exclude visual-only candidates')
    parser.add_argument('--replacement', action='store_true',
                        help='Text/image/icon replacement candidates with presentation review')
    parser.add_argument('--gt-review', type=Path,
                        help='JSON: {"000": {"original_gt_index": {review fields}}}')
    parser.add_argument('--config', type=Path, default=REPO/'config.local.json')
    parser.add_argument('--cache-root', type=Path,
                        help='Reuse <ID>/inference.json only, never geometry matching')
    parser.add_argument('--execute', action='store_true', help='Otherwise preflight only')
    parser.add_argument('--output-root', type=Path, required=True,
                        help='New directory under content/results; refuses overwrite')
    args = parser.parse_args(argv)
    if args.text_only and args.replacement:
        parser.error('Choose either --text-only or --replacement')
    if args.gt_review and not args.replacement:
        parser.error('--gt-review requires --replacement')
    reviews = c.read(args.gt_review) if args.gt_review else {}
    if not isinstance(reviews, dict):
        parser.error('GT review must be a JSON object keyed by zero-padded pair ID')
    out = args.output_root.resolve()
    if not out.is_relative_to(ROOT/'content/results') or out == ROOT/'content/results':
        parser.error('Output must be a new child directory of content/results')
    if args.count <= 0 or args.start_id < 0:
        parser.error('Invalid ID range')
    if out.exists():
        parser.error('Output exists; choose a new run directory')
    config = c.read(args.config) if args.config.exists() else {}
    defaults = {'dataset1': 'dataset1', 'baseline': 'output_dataset1/baseline/json',
                'inject': 'output_dataset1/inject/json', 'groundtruth': 'output_dataset1/groundtruth'}
    paths = {k: (REPO/Path(config.get(k, v))).resolve() for k, v in defaults.items()}
    indices = {k: numeric_index(v, '*.json' if k != 'dataset1' else '*_*') for k, v in paths.items()}
    cache_index = numeric_index(args.cache_root.resolve(), '*') if args.cache_root else {}
    records, issues, hashes = [], [], []
    if args.gt_review:
        hashes.append({'path': str(args.gt_review.resolve()), 'sha256': digest(args.gt_review)})
    for number in range(args.start_id, args.start_id+args.count):
        pid = f'{number:03d}'
        mapped = {k: unique(index, number) for k, index in indices.items()}
        errors = [f'{k}: expected one source, found {len(indices[k].get(number, []))}'
                  for k, value in mapped.items() if value is None]
        record = {'pair_id': pid, 'sources': {k: str(v) if v else None for k, v in mapped.items()},
                  'content_gt': None, 'errors': errors}
        if mapped['groundtruth']:
            try:
                record['content_gt'] = normalize_gt(c.read(mapped['groundtruth']))
                if args.replacement:
                    from content.replacement_changes import audit_gt
                    record['gt_audit'] = audit_gt(record['content_gt'], reviews.get(pid, {}))
            except (ValueError, KeyError, TypeError) as error:
                errors.append(str(error))
        if all(mapped[k] for k in ('baseline', 'inject', 'dataset1')):
            b, a, image = (mapped[k] for k in ('baseline', 'inject', 'dataset1'))
            if b.name != a.name or b.stem.removesuffix('_full_hierarchy') != image.name:
                errors.append('Source stem mismatch')
            for side, filename in [('before_image', 'baseline.png'), ('after_image', 'inject.png')]:
                path = image/filename
                record['sources'][side] = str(path)
                if not path.is_file():
                    errors.append(f'Missing {filename}')
        if args.cache_root:
            folder = unique(cache_index, number)
            cache = folder/'inference.json' if folder else None
            record['cache'] = str(cache) if cache else None
            if not cache or not cache.is_file():
                errors.append('Missing/ambiguous inference cache')
        for value in record['sources'].values():
            if value and Path(value).is_file():
                hashes.append({'path': value, 'sha256': digest(Path(value))})
        if record.get('cache') and Path(record['cache']).is_file():
            hashes.append({'path': record['cache'], 'sha256': digest(Path(record['cache']))})
        records.append(record)
        if errors:
            issues.append({'pair_id': pid, 'errors': errors})
    if set(reviews) - {r['pair_id'] for r in records}:
        parser.error('GT review contains sample IDs outside requested cohort')
    out.mkdir(parents=True)
    manifest = {'cohort_rule': 'inclusive numeric start; count consecutive IDs; never skip missing IDs',
                'start_id': args.start_id, 'count': args.count,
                'resolved_roots': {k: str(v) for k, v in paths.items()},
                'samples': records, 'inputs': hashes,
                'code_sha256': {str(p.relative_to(CODE_ROOT)): digest(p) for p in
                                [CODE_ROOT/'content/scripts/evaluate_first500.py', CODE_ROOT/'content/pipeline.py', CODE_ROOT/'content/text_changes.py', CODE_ROOT/'content/replacement_changes.py', CODE_ROOT/'shared/matching.py', CODE_ROOT/'shared/regions.py']},
                'text_only': args.text_only, 'replacement': args.replacement}
    c.write(out/'input_manifest.json', manifest)
    c.write(out/'config.json', c.CONFIG)
    if args.replacement:
        from content.replacement_changes import CONFIG as replacement_config
        c.write(out/'replacement_config.json', replacement_config)
        template = {r['pair_id']: {str(t['original_gt_index']): {
            'content_type': 'unknown', 'content_replaced': None, 'same_element': None,
            'presentation_unchanged': None, 'unintentional_confirmed': None,
            'evidence': ''} for t in r['content_gt']} for r in records if r['content_gt']}
        c.write(out/'gt_review_template.json', template)
        c.write(out/'gt_audit.json', {r['pair_id']: r.get('gt_audit', []) for r in records})
    preflight = {'status': 'blocked_missing_or_invalid_inputs' if issues else 'ready',
                 'requested_samples': args.count, 'samples_with_errors': len(issues),
                 'available_gt_files': sum(r['content_gt'] is not None for r in records),
                 'content_samples': sum(bool(r['content_gt']) for r in records),
                 'content_gt': sum(len(r['content_gt'] or []) for r in records), 'issues': issues}
    c.write(out/'preflight.json', preflight)
    c.write(out/'run_status.json', {'status': preflight['status'] if issues else 'preflight_only'})
    print(json.dumps({k: v for k, v in preflight.items() if k != 'issues'}, indent=2))
    if not args.execute or issues:
        (out/'REPORT.md').write_text(
            '# Content evaluation preflight\n\n'
            f'Status: {preflight["status"]}; requested samples: {args.count}; '
            f'samples with input errors: {len(issues)}.\n\n'
            'No inference or scoring was performed. Missing GT does not mean no content changes. '
            'See preflight.json for each ID and input_manifest.json for resolved roots.\n', encoding='utf-8')
        return 2 if issues else 0
    # Phase 1: all cohort predictions written before the first evaluation call.
    c.write(out/'run_status.json', {'status': 'inference_without_gt'})
    frozen = []
    for record in records:
        src = record['sources']; pid = record['pair_id']
        result = c.read(record['cache']) if args.cache_root else c.infer(
            src['baseline'], src['inject'], src['before_image'], src['after_image'], dict(c.CONFIG))
        if result.get('uses_groundtruth') is not False or result.get('config') != c.CONFIG:
            raise ValueError(f'{pid}: incompatible or non-GT-free cache; use fresh inference')
        if result.get('text_profile') or result.get('replacement_profile'):
            raise ValueError(f'{pid}: use original content-v1 inference cache, not an already projected text profile')
        if args.text_only:
            from content.text_changes import select_text_changes
            result = select_text_changes(result)
        if args.replacement:
            from content.replacement_changes import select_replacements
            result = select_replacements(result)
        with c.Image.open(src['before_image']) as bi, c.Image.open(src['after_image']) as ai:
            if bi.size != ai.size or list(bi.size) != result['image_size']:
                raise ValueError(f'{pid}: viewport mismatch')
        for target in record['content_gt']:
            x0, y0, x1, y1 = target['xyxy']
            if x0 < 0 or y0 < 0 or x1 > result['image_size'][0] or y1 > result['image_size'][1]:
                raise ValueError(f'{pid}: GT outside viewport; review coordinates, never silently clip')
        c.write(out/pid/'inference.json', result)
        frozen.append({'pair_id': pid, 'sha256': digest(out/pid/'inference.json')})
        print(f'{pid}: {len(result["predictions"])} predictions', flush=True)
    c.write(out/'prediction_manifest.json', {'phase': 'all_predictions_frozen_before_scoring',
            'cache_input_provenance': 'not independently verified' if args.cache_root else 'fresh inference from hashed inputs',
            'artifacts': frozen})
    c.write(out/'run_status.json', {'status': 'joining_gt'})
    sample_rows, gt_rows, prediction_rows, candidate_rows = [], [], [], []
    reviewed_evaluations = []
    for record in records:
        pid = record['pair_id']; result = c.read(out/pid/'inference.json')
        gt = c.read(record['sources']['groundtruth'])
        c.write(out/pid/'normalized_gt.json', record['content_gt'])
        evaluation = c.evaluate(result, gt)
        c.write(out/pid/'evaluation.json', evaluation)
        if args.replacement:
            from content.replacement_changes import evaluate_reviewed
            reviewed = evaluate_reviewed(result, gt, record['gt_audit'])
            c.write(out/pid/'reviewed_pure_gt_evaluation.json', reviewed)
            reviewed_evaluations.append((record, result, reviewed))
            for correspondence in ('matched_pairs', 'uncertain_pairs'):
                for pair in result[correspondence]:
                    candidate_rows.append({'pair_id': pid, 'correspondence': correspondence,
                        'before_id': pair['before_id'], 'after_id': pair['after_id'],
                        'before_bbox': json.dumps(pair['before_bbox']), 'after_bbox': json.dumps(pair['after_bbox']),
                        **pair['replacement'], 'reason': pair['content']['reason'],
                        'human_content_replaced': '', 'human_presentation_unchanged': '',
                        'human_same_element': '', 'human_unintentional_confirmed': '', 'human_evidence': ''})
        sample_rows.append({'pair_id': pid, 'gt_count': evaluation['gt_count'],
                            'covered': evaluation['covered'], 'ocr_supported_covered': evaluation['ocr_supported_covered'],
                            'positive_sample_pass': evaluation['pass'] if evaluation['gt_count'] else None,
                            'predictions': evaluation['predictions'], 'unverified_predictions': evaluation['unverified_predictions']})
        for target in evaluation['gt_coverage']:
            audited = next((t for t in record.get('gt_audit', []) if t['original_gt_index'] == target['original_gt_index']), {})
            gt_rows.append({'pair_id': pid, 'original_gt_index': target['original_gt_index'],
                            'covered': target['covered'], 'fallback': target['gt'].get('fallback', False),
                            'labels': json.dumps(target['gt']['type']), 'links': json.dumps(target['links']),
                            'diagnostic_hypotheses': '|'.join(target['diagnostic_hypotheses']),
                            'gt_eligibility': audited.get('eligibility', 'not_audited'),
                            'reviewed_content_type': audited.get('reviewed_content_type', 'unknown'), 'human_cause': ''})
        for i, prediction in enumerate(result['predictions']):
            linked = [t['original_gt_index'] for t in evaluation['gt_coverage'] if any(i in link['prediction_indices'] for link in t['links'])]
            prediction_rows.append({'pair_id': pid, 'prediction_index': i,
                                    'before_id': prediction['before_id'], 'after_id': prediction['after_id'],
                                    'kind': prediction['content']['kind'],
                                    'before_text': prediction['content']['before_text'], 'after_text': prediction['content']['after_text'],
                                    'linked_gt_indices': json.dumps(linked),
                                    'human_semantic_correct': '', 'human_correspondence_correct': ''})
    for entry in hashes:
        if digest(Path(entry['path'])) != entry['sha256']:
            raise ValueError(f'Input changed during run: {entry["path"]}')
    for entry in frozen:
        if digest(out/entry['pair_id']/'inference.json') != entry['sha256']:
            raise ValueError('Frozen inference changed during evaluation')
    positives = [r for r in sample_rows if r['gt_count']]
    denominator = sum(r['gt_count'] for r in positives)
    covered = sum(r['covered'] for r in positives)
    summary = {'samples': len(records), 'content_positive_samples': len(positives),
               'prediction_profile': 'content-replacement-v1' if args.replacement else 'visible-text-v1' if args.text_only else 'content-v1',
               'gt_count': denominator, 'covered': covered,
               'coverage': covered/denominator if denominator else None,
               'pass_positive_samples': sum(r['positive_sample_pass'] for r in positives),
               'ocr_supported_covered': sum(r['ocr_supported_covered'] for r in positives),
               'predictions': len(prediction_rows),
               'unverified_predictions': sum(r['unverified_predictions'] for r in sample_rows),
               'predictions_on_samples_without_content_gt': sum(r['predictions'] for r in sample_rows if not r['gt_count']),
               'protocol': 'endpoint union: gt_fraction>=0.50 AND support_fraction>=0.25; many-to-many',
               'not_precision_recall_f1': True, 'independent_test': False}
    for name, rows in [('samples.csv', sample_rows), ('gt_join.csv', gt_rows), ('prediction_review.csv', prediction_rows)]:
        save_csv(out/name, rows)
    if args.replacement:
        counts = Counter(t['eligibility'] for r in records for t in r.get('gt_audit', []))
        by_type = {}
        for kind in ('text', 'image', 'icon', 'unknown'):
            targets = [(t, ev) for record, _, ev in reviewed_evaluations for t in record['gt_audit']
                       if t['eligibility'] == 'pure' and t['reviewed_content_type'] == kind]
            hit = sum(any(g['original_gt_index'] == t['original_gt_index'] and g['covered'] for g in ev['gt_coverage']) for t, ev in targets)
            by_type[kind] = {'reviewed_pure_gt': len(targets), 'candidate_covered': hit,
                             'coverage': hit/len(targets) if targets else None}
        reviewed_count = sum(ev['gt_count'] for _, _, ev in reviewed_evaluations)
        reviewed_covered = sum(ev['covered'] for _, _, ev in reviewed_evaluations)
        summary.update(metric='provisional_candidate_localization_on_original_content_gt',
                       gt_eligibility_counts=dict(counts), by_reviewed_content_type=by_type,
                       candidate_type_hints=dict(Counter(p['replacement']['content_type_hint'] for _, r, _ in reviewed_evaluations for p in r['predictions'])),
                       reviewed_pure_gt=reviewed_count, reviewed_pure_gt_candidate_covered=reviewed_covered,
                       reviewed_pure_gt_candidate_coverage=reviewed_covered/reviewed_count if reviewed_count else None,
                       semantic_correctness_confirmed=False, confirmed_content_changes=None)
        save_csv(out/'candidate_review.csv', candidate_rows)
        (out/'REPORT.md').write_text(
            '# Content replacement audit\n\n'
            f'Profile: content-replacement-v1; samples: {len(records)}.\n\n'
            f'Provisional candidate localization on original GT: {covered}/{denominator}.\n\n'
            f'Reviewed pure GT candidate localization: {reviewed_covered}/{reviewed_count}; '
            'undefined when denominator is zero.\n\n'
            'Text/image/icon candidates are not confirmed replacements. Font/color/style and '
            'unintentional replacement require review. See gt_review_template.json, gt_join.csv, '
            'candidate_review.csv, and summary.json. Never report this as semantic precision/recall.\n', encoding='utf-8')
    c.write(out/'summary.json', summary)
    c.write(out/'validation.json', {'input_hashes_verified': True, 'prediction_hashes_verified': True,
                                   'all_requested_samples_evaluated': len(sample_rows) == args.count})
    c.write(out/'run_status.json', {'status': 'complete'})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
