"""Audit direct text/GT coverage using saved detections and image evidence."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from content import pipeline as c
from content.scripts.evaluate_first500 import digest
from geometry.pipeline import bbox_iou_xywh
from shared.regions import area, intersect, union_area


def region_diagnostics(gt, elements):
    box = gt['xyxy']
    overlap = []
    for element in elements:
        intersection = area(intersect(box, element['bbox']))
        if intersection <= 0:
            continue
        iou, ccr = bbox_iou_xywh(gt['original'], element['bbox'])
        overlap.append({'id': element['id'], 'bbox': element['bbox'], 'text': element['class_label'],
                        'iou': iou, 'ccr': ccr, 'intersection_fraction_of_detection': intersection/area(element['bbox']),
                        'detection_to_gt_area_ratio': area(element['bbox'])/area(box)})
    best = max(overlap, key=lambda e: max(e['iou']/.3, e['ccr']/.5), default=None)
    union_fraction = union_area([intersect(e['bbox'], box) for e in overlap])/area(box)
    covered = any(e['iou']+1e-12 >= .3 or e['ccr']+1e-12 >= .5 for e in overlap)
    reason = 'covered' if covered else 'no_text_overlap' if not overlap else 'fragmented_text_union_reaches_50pct' if union_fraction >= .5 else 'text_overlap_below_threshold'
    return {'covered': covered, 'diagnostic': reason, 'best_overlap': best,
            'overlapping_text_boxes': len(overlap), 'text_union_gt_fraction': union_fraction,
            'links': overlap}


def contact_sheets(out, rows, samples, limit=5):
    pages = []
    for offset in range(0, len(rows), limit):
        page_rows = rows[offset:offset+limit]
        canvas = Image.new('RGB', (1040, 220*len(page_rows)), 'white')
        draw = ImageDraw.Draw(canvas)
        for row_number, row in enumerate(page_rows):
            y = row_number*220
            pid = row['pair_id']
            draw.text((8, y+5), f'{pid} GT {row["original_gt_index"]} {row["original"].get("selector", "")[:100]}', fill='black')
            for column, side in enumerate(('baseline', 'inject')):
                key = 'before_image' if side == 'baseline' else 'after_image'
                with Image.open(samples[pid]['sources'][key]) as im:
                    im = im.convert('RGB')
                painter = ImageDraw.Draw(im)
                for element in row['diagnostics'][side]['links']:
                    painter.rectangle(element['bbox'], outline='cyan', width=2)
                painter.rectangle(row['xyxy'], outline='red', width=2)
                b = row['xyxy']
                crop = im.crop((max(0, b[0]-24), max(0, b[1]-24), min(im.width, b[2]+24), min(im.height, b[3]+24)))
                crop.thumbnail((510, 165))
                x = column*520
                canvas.paste(crop, (x+(510-crop.width)//2, y+42))
                d = row['diagnostics'][side]
                best = d['best_overlap']
                detail = f'IoU={best["iou"]:.3f} CCR={best["ccr"]:.3f}' if best else 'no text overlap'
                draw.text((x+8, y+22), f'{side} {detail}', fill='black')
        path = out/f'missed_review_{offset//limit+1}.png'
        canvas.save(path)
        pages.append(path.name)
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    out = args.output_root.resolve()
    if out.exists() or not out.is_relative_to(root) or out == root:
        parser.error('Choose a new audit subdirectory under the detection coverage run')
    provenance = c.read(root/'manifest.json')
    for source in provenance['inputs']:
        if digest(Path(source['path'])) != source['sha256']:
            raise ValueError(f'Input differs from scored run: {source["path"]}')
    samples = {s['pair_id']: s for s in c.read(provenance['input_manifest'])['samples']}
    rows = c.read(root/'gt_join.json')
    counts = {side: Counter() for side in ('baseline', 'inject')}
    populations = {side: Counter() for side in counts}
    strata = defaultdict(lambda: Counter())
    oversized = {side: [] for side in counts}
    all_detection = {}
    for pid, sample in samples.items():
        all_detection[pid] = {}
        for side in counts:
            raw = c.read(sample['sources'][side])['elements']
            all_detection[pid][side] = raw
            populations[side].update(e.get('source', 'unknown') for e in raw)
            populations[side]['non_ocr_with_ocr_provenance'] += sum(e.get('source') != 'ocr' and 'ocr' in (e.get('sources') or []) for e in raw)
    for row in rows:
        pid = row['pair_id']
        selected = c.read(root/pid/'text_detections.json')
        row['diagnostics'] = {side: region_diagnostics(row, selected[side]) for side in counts}
        selector = row['original'].get('selector', '')
        stratum = 'image_selector' if selector.startswith('img') else 'input_selector' if selector.startswith('input') else 'other_selector'
        strata[stratum]['gt'] += 1
        for side in counts:
            d = row['diagnostics'][side]
            assert d['covered'] == row[side+'_covered']
            counts[side][d['diagnostic']] += 1
            strata[stratum][side+'_covered'] += int(d['covered'])
            links = row['links'][side]
            if links and all(l['iou'] < .3 for l in links):
                counts[side]['covered_by_ccr_only'] += 1
            if links and all(area(l['bbox'])/area(row['xyxy']) > 4 for l in links):
                oversized[side].append({'pair_id': pid, 'gt_index': row['original_gt_index']})
            nontext = [e for e in all_detection[pid][side] if e.get('source') != 'ocr' and len(e.get('bbox', [])) == 4 and area(e['bbox']) > 0]
            row['diagnostics'][side]['nontext_box_covers_gt'] = any(
                iou+1e-12 >= .3 or ccr+1e-12 >= .5 for iou, ccr in
                (bbox_iou_xywh(row['original'], e['bbox']) for e in nontext))
    sweep = {}
    for threshold in (.10, .20, .30, .50):
        sweep[str(threshold)] = {side: sum(any(e['iou']+1e-12 >= threshold or e['ccr']+1e-12 >= .50 for e in r['diagnostics'][side]['links']) for r in rows) for side in counts}
    missed = [r for r in rows if not r['either_covered']]
    summary = {'gt': len(rows), 'side_diagnostics': counts, 'detection_sources': populations,
               'strata': strata, 'oversized_only_gt_links': oversized, 'iou_sensitivity_with_fixed_ccr_0.5': sweep,
               'both_missed': len(missed),
               'both_missed_with_some_text_overlap': sum(any(r['diagnostics'][s]['overlapping_text_boxes'] for s in counts) for r in missed),
               'both_missed_rescued_by_text_union': sum(any(r['diagnostics'][s]['text_union_gt_fraction'] >= .5 for s in counts) for r in missed),
               'baseline_only': sum(r['baseline_covered'] and not r['inject_covered'] for r in rows),
               'inject_only': sum(r['inject_covered'] and not r['baseline_covered'] for r in rows)}
    sensitivity = {}
    for mode in ('iou_only', 'with_support25'):
        hits = {}
        for side in counts:
            hits[side] = []
            for row in rows:
                links = row['diagnostics'][side]['links']
                if mode == 'iou_only':
                    covered = any(e['iou']+1e-12 >= .3 for e in links)
                else:
                    covered = any((e['iou']+1e-12 >= .3 or e['ccr']+1e-12 >= .5)
                                  and e['intersection_fraction_of_detection']+1e-12 >= .25 for e in links)
                hits[side].append(covered)
        sensitivity[mode] = {side: sum(values) for side, values in hits.items()}
        sensitivity[mode]['either'] = sum(a or b for a, b in zip(hits['baseline'], hits['inject']))
        sensitivity[mode]['both'] = sum(a and b for a, b in zip(hits['baseline'], hits['inject']))
    summary['metric_sensitivity'] = sensitivity
    out.mkdir(parents=True)
    c.write(out/'summary.json', summary)
    c.write(out/'gt_diagnostics.json', rows)
    c.write(out/'both_missed_analysis.json', missed)
    pages = contact_sheets(out, missed, samples)
    c.write(out/'validation.json', {'source_hashes_verified': True, 'baseline_scores_reproduced': True,
                                   'gt_count': len(rows), 'review_images': pages,
                                   'audit_code_sha256': digest(Path(__file__))})
    print(__import__('json').dumps(summary, indent=2))


if __name__ == '__main__':
    main()
