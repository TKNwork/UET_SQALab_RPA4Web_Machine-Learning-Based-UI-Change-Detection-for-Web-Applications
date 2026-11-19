"""Compare geometry and OCR text changes from one pair of full hierarchies."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from content import pipeline as content
from content.hierarchy_text import normalize_hierarchy_text
from content.scripts.evaluate_text_pairs import select_predictions
from geometry import pipeline as geometry
from geometry.prediction_groups import build_prediction_groups
from shared.matching import normalize_hierarchy
from visibility import candidates as visibility_candidates


def pair_key(record):
    return record['before_id'], record['after_id']


def merge_events(geometry_changes, content_changes):
    events = {}
    for kind, changes in [('geometry', geometry_changes), ('content', content_changes)]:
        for record in changes:
            event = events.setdefault(pair_key(record), {
                field: record[field]
                for field in ('before_id', 'after_id', 'before_bbox', 'after_bbox')
            })
            event[kind] = record
    return [{'id': index, **event} for index, event in enumerate(events.values(), 1)]


def compare(before, after, before_image, after_image, *, reuse_geometry=False):
    paths = (before, after)
    data = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    normalized = [normalize_hierarchy(value, path) for path, value in zip(paths, data)]
    ocr = {path: normalize_hierarchy_text(value, path) for path, value in zip(paths, data)}
    image_paths = {'before': before_image, 'after': after_image}
    with Image.open(before_image) as source:
        before_rgb = source.convert('RGB')
    with Image.open(after_image) as source:
        after_rgb = source.convert('RGB')
    images = before_rgb, after_rgb
    geo = geometry.infer_detections(
        before.stem.removesuffix('_full_hierarchy'), *normalized, image_paths,
        geometry.DEFAULT_CONFIG, images=images
    )
    text = select_predictions(content.infer(
        *paths, before_image, after_image, ocr_only=True,
        detection_loader=ocr.__getitem__, images=images,
        geometry_pairs=geo['matched_pairs'] if reuse_geometry else None
    ), True)
    changes = [record for record in geo['matched_pairs']
               if geometry.classify_change(record['delta'], 'edges8')]
    return {
        'schema_version': 1,
        'geometry': geo,
        'geometry_groups': build_prediction_groups(changes, geo['matched_pairs'], before_rgb.size),
        'content': text,
        'visibility_candidates': visibility_candidates(geo, text),
        'events': merge_events(changes, text['predictions']),
    }


def render(result, target):
    """Draw solid blue G boxes and dashed orange C boxes on both screenshots."""
    font = ImageFont.load_default(size=19)
    panels = []
    for side, title in [('before', 'BEFORE'), ('after', 'AFTER')]:
        with Image.open(result['geometry']['image_paths'][side]) as image:
            panel = Image.new('RGB', (image.width, image.height + 76), '#111827')
            panel.paste(image, (0, 76))
        draw = ImageDraw.Draw(panel)
        draw.text((12, 8), title, fill='white', font=font)
        draw.text((12, 35), 'G = Geometry (solid)   C = Content (dashed)', fill='white', font=font)
        for event in result['events']:
            for kind, letter, color in [('geometry', 'G', '#1263f2'), ('content', 'C', '#cf4300')]:
                if kind not in event:
                    continue
                x1, y1, x2, y2 = map(round, event[kind][side + '_bbox'])
                y1, y2 = y1 + 76, y2 + 76
                offset = 5 if kind == 'content' and 'geometry' in event else 0
                x1, y1, x2, y2 = x1 + offset, y1 + offset, x2 - offset, y2 - offset
                if x2 < x1 or y2 < y1:
                    continue
                if kind == 'geometry':
                    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                else:
                    for x in range(x1, x2 + 1, 12):
                        for y in (y1, y2):
                            draw.line((x, y, min(x + 7, x2), y), fill=color, width=3)
                    for y in range(y1, y2 + 1, 12):
                        for x in (x1, x2):
                            draw.line((x, y, x, min(y + 7, y2)), fill=color, width=3)
                label = f"{letter}{event['id']}"
                label_y = max(54, y1 - 23) if kind == 'geometry' else y2
                draw.text((max(0, x1) + 3, label_y), label, font=font, fill=color,
                          stroke_width=2, stroke_fill='white')
        panels.append(panel)
    left, right = panels
    canvas = Image.new('RGB', (left.width + right.width + 12, max(left.height, right.height)), '#111827')
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + 12, 0))
    canvas.save(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('before', 'after', 'before-image', 'after-image', 'output'):
        parser.add_argument(f'--{arg}', required=True, type=Path)
    parser.add_argument('--visualization', type=Path)
    parser.add_argument('--reuse-geometry', action='store_true', help='Try geometry matches before OCR assignment')
    args = parser.parse_args()
    result = compare(args.before, args.after, args.before_image, args.after_image,
                     reuse_geometry=args.reuse_geometry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.visualization:
        args.visualization.parent.mkdir(parents=True, exist_ok=True)
        render(result, args.visualization)


if __name__ == '__main__':
    main()
