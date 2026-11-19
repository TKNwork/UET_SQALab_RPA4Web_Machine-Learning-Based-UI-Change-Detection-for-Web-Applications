"""Direct text-detection coverage of content GT, independently on each side."""
import argparse
from collections import Counter
from pathlib import Path

from content import pipeline as content
from content.scripts.evaluate_first500 import digest, normalize_gt
from geometry.pipeline import bbox_iou_xywh
from shared.matching import load_normalized

ROOT = Path(__file__).resolve().parents[2]


def evaluate_text_boxes(targets, before, after):
    rows = []
    for target in targets:
        row = {**target, 'links': {}}
        for side, elements in [('baseline', before), ('inject', after)]:
            links = []
            for element in elements:
                iou, ccr = bbox_iou_xywh(target['original'], element['bbox'])
                if iou + 1e-12 >= .30 or ccr + 1e-12 >= .50:
                    links.append({'detection_id': element['id'], 'bbox': element['bbox'],
                                  'text': element['class_label'], 'iou': iou, 'ccr': ccr})
            row['links'][side] = links
        row['baseline_covered'] = bool(row['links']['baseline'])
        row['inject_covered'] = bool(row['links']['inject'])
        row['either_covered'] = row['baseline_covered'] or row['inject_covered']
        row['both_covered'] = row['baseline_covered'] and row['inject_covered']
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-manifest', type=Path, required=True,
                        help='Source manifest from the bounded cohort preflight')
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root.resolve()
    if not out.is_relative_to(ROOT/'content/results') or out == ROOT/'content/results' or out.exists():
        parser.error('Choose a new child directory under content/results')
    manifest = content.read(args.input_manifest)
    samples = manifest['samples']
    if not samples or any(sample['errors'] for sample in samples):
        parser.error('Source manifest must contain a complete cohort')
    out.mkdir(parents=True)
    content.write(out/'run_status.json', {'status': 'extracting_and_matching'})
    input_hashes = []
    summaries = []
    all_rows = []
    for sample in samples:
        pid, sources = sample['pair_id'], sample['sources']
        detections = {}
        warnings = {}
        for side, source in [('baseline', 'baseline'), ('inject', 'inject')]:
            path = Path(sources[source])
            input_hashes.append({'path': str(path), 'sha256': digest(path)})
            elements, warnings[side] = load_normalized(path)
            # OCR provenance identifies text elements, even when recognition is empty.
            detections[side] = [e for e in elements if e['source'] == 'ocr']
        gt_path = Path(sources['groundtruth'])
        input_hashes.append({'path': str(gt_path), 'sha256': digest(gt_path)})
        targets = normalize_gt(content.read(gt_path))
        rows = evaluate_text_boxes(targets, detections['baseline'], detections['inject'])
        for row in rows:
            row['pair_id'] = pid
        all_rows.extend(rows)
        summary = {'pair_id': pid, 'gt_count': len(rows),
                   'normalization_warnings': warnings,
                   'baseline_text_elements': len(detections['baseline']),
                   'inject_text_elements': len(detections['inject'])}
        for mode in ('baseline', 'inject', 'either', 'both'):
            summary[mode+'_covered'] = sum(r[mode+'_covered'] for r in rows)
            summary[mode+'_pass'] = all(r[mode+'_covered'] for r in rows) if rows else None
        summaries.append(summary)
        content.write(out/pid/'text_detections.json', detections)
        content.write(out/pid/'normalization_warnings.json', warnings)
        content.write(out/pid/'gt_join.json', rows)
    for source in input_hashes:
        if digest(Path(source['path'])) != source['sha256']:
            raise ValueError(f'Input changed during evaluation: {source["path"]}')
    total = len(all_rows)
    result = {'samples': len(samples), 'content_positive_samples': sum(bool(s['gt_count']) for s in summaries),
              'content_gt': total, 'text_selection': 'source == ocr; preserve original boxes and labels',
              'rule': 'singleton detection IoU>=0.30 OR CCR=intersection/GT_area>=0.50',
              'multilabel_gt': dict(Counter('+'.join(sorted(r['original']['type'])) for r in all_rows)),
              'baseline_text_elements': sum(s['baseline_text_elements'] for s in summaries),
              'inject_text_elements': sum(s['inject_text_elements'] for s in summaries)}
    result['normalization_warning_count'] = sum(len(w) for s in summaries for w in s['normalization_warnings'].values())
    result['samples_with_normalization_warnings'] = [s['pair_id'] for s in summaries if any(s['normalization_warnings'].values())]
    for mode in ('baseline', 'inject', 'either', 'both'):
        covered = sum(r[mode+'_covered'] for r in all_rows)
        result[mode] = {'covered': covered, 'missed': total-covered,
                        'coverage': covered/total if total else None,
                        'pass_positive_samples': sum(s[mode+'_pass'] is True for s in summaries)}
    content.write(out/'summary.json', result)
    content.write(out/'samples.json', summaries)
    content.write(out/'gt_join.json', all_rows)
    content.write(out/'missed_gt.json', [r for r in all_rows if not r['either_covered']])
    content.write(out/'manifest.json', {'input_manifest': str(args.input_manifest.resolve()),
                  'input_manifest_sha256': digest(args.input_manifest), 'inputs': input_hashes,
                  'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                      [Path(__file__), ROOT/'shared/matching.py', ROOT/'geometry/pipeline.py',
                       ROOT/'content/scripts/evaluate_first500.py']}, 'input_hashes_verified': True})
    lines = ['# Text detection coverage against content GT', '',
             f'{len(samples)} samples, {result["content_positive_samples"]} content-positive samples, {total} GT.', '',
             'Text elements are selected by OCR provenance and matched spatially to original content GT.', '',
             'Rule inherited from geometry: singleton IoU >= 0.30 OR intersection / GT area >= 0.50.', '',
             '| Side | Covered | Coverage | Fully covered positive samples |',
             '|---|---:|---:|---:|']
    for mode in ('baseline', 'inject', 'either', 'both'):
        value = result[mode]
        fraction = f'{value["coverage"]:.2%}' if total else 'undefined'
        lines.append(f'| {mode} | {value["covered"]}/{total} | {fraction} | {value["pass_positive_samples"]} |')
    lines += ['', 'Either means at least one side has a qualifying text box. Both checks the same flat GT region on each side independently.',
              f'Normalization skipped malformed detection records with {result["normalization_warning_count"]} warnings. Details are retained per sample.',
              'Every qualifying detection link is retained. Flat GT provides no endpoint identity.',
              'See gt_join.json and missed_gt.json for per-region evidence.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    content.write(out/'run_status.json', {'status': 'complete'})
    print(__import__('json').dumps(result, indent=2))


if __name__ == '__main__':
    main()
