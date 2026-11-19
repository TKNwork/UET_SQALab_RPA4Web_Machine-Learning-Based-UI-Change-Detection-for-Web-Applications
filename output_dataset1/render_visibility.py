"""Render one scored sample with candidate and ground-truth boxes."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render(row, images, output):
    font = ImageFont.load_default(size=20)
    panels = []
    for side, filename, label in [('before', 'baseline.png', 'V-'), ('after', 'inject.png', 'V+')]:
        with Image.open(images / row['sample'] / filename) as source:
            panel = Image.new('RGB', (source.width, source.height + 56), '#131827')
            panel.paste(source, (0, 56))
        draw = ImageDraw.Draw(panel)
        draw.text((12, 12), f'{side.upper()}   Purple: candidate   Green: GT', fill='white', font=font)
        for items, key, prefix, color in [
            (row['targets'], 'xyxy', 'GT', '#13823a'),
            (row['candidates'], 'bbox', label, '#7c26c7'),
        ]:
            for i, item in enumerate(items):
                if 'side' in item and item['side'] != side:
                    continue
                x1, y1, x2, y2 = item[key]
                inset = 0 if prefix == 'GT' else 3
                draw.rectangle((x1-inset, y1+56-inset, x2+inset, y2+56+inset), outline=color, width=3)
                y = y1 + 32 if prefix == 'GT' else y2 + 57
                draw.text((x1, y), f'{prefix}{i+1}', font=font, fill=color,
                          stroke_fill='white', stroke_width=2)
        panels.append(panel)
    left, right = panels
    canvas = Image.new('RGB', (left.width + right.width + 8, max(left.height, right.height)), '#131827')
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + 8, 0))
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores', type=Path)
    parser.add_argument('images', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--sample')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.scores.read_text(encoding='utf-8').splitlines()]
    if args.sample:
        row = next(row for row in rows if row['sample'] == args.sample)
    else:
        row = min((row for row in rows if row['matches']),
                  key=lambda row: len(row['candidates']) + len(row['targets']))
    render(row, args.images, args.output)
    print(row['sample'])


if __name__ == '__main__':
    main()
