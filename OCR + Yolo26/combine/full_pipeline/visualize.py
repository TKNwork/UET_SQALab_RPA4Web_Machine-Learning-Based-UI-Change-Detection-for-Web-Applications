import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'yolo_pixelweb_dataset', 'images', 'val')
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'yolo_pixelweb_dataset', 'images', 'val')

MATCH_RESULTS_DIR = os.path.join(BASE_DIR, 'combine', 'full_pipeline', 'results', 'json')
OUTPUT_VIS_DIR = os.path.join(BASE_DIR, 'combine', 'full_pipeline', 'results', 'visualized')

COLOR_SOURCE = {
    'ocr': (0, 255, 0),        # Green
    'small': (255, 0, 0),      # Red
    'medium': (0, 100, 255),   # Blue
    'large': (0, 150, 0),      # Dark Green
    'cv': (147, 112, 219)      # Purple
}


def draw_legend(draw, font):
    legend_box = [10, 10, 270, 140]
    draw.rectangle(legend_box, fill=(0, 0, 0), outline=(255, 255, 255), width=1)

    draw.text((15, 14), "Full Pipeline Hierarchy Legend:", fill=(255, 255, 255), font=font)

    items = [
        ("OCR Text", COLOR_SOURCE['ocr']),
        ("Small YOLO", COLOR_SOURCE['small']),
        ("Medium YOLO", COLOR_SOURCE['medium']),
        ("Large YOLO", COLOR_SOURCE['large']),
        ("CV Contour", COLOR_SOURCE['cv'])
    ]

    y_offset = 32
    for label, color in items:
        sample_box = [15, y_offset + 2, 35, y_offset + 10]
        draw.rectangle(sample_box, outline=color, width=2)
        draw.text((42, y_offset - 2), label, fill=color, font=font)
        y_offset += 20


def visualize_full_hierarchy(image_name: str) -> str:
    clean_id = image_name.replace('.png', '').replace('_full_hierarchy', '')

    img_path = os.path.join(VAL_IMAGES_DIR, f"{clean_id}.png")
    match_json_path = os.path.join(MATCH_RESULTS_DIR, f"{clean_id}_full_hierarchy.json")

    if not os.path.exists(img_path) or not os.path.exists(match_json_path):
        return ""

    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    with open(match_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    elements = data.get('elements', [])
    elem_dict = {el['id']: el for el in elements}

    # 1. Sắp xếp elements theo diện tích giảm dần (vẽ Large trước, Small/OCR sau)
    elements_sorted = sorted(elements, key=lambda x: x['area'], reverse=True)

    for el in elements_sorted:
        bbox = el['bbox']
        src = el['source']
        color = COLOR_SOURCE.get(src, (255, 255, 255))
        draw.rectangle(bbox, outline=color, width=1 if src in ('ocr', 'cv') else 2)

    # 2. Vẽ đường nối từ Child tới Primary Parent
    for el in elements:
        parent_info = el.get('primary_parent')
        if parent_info:
            pid = parent_info['id']
            if pid in elem_dict:
                cb = el['bbox']
                pb = elem_dict[pid]['bbox']
                cc = ((cb[0] + cb[2]) / 2.0, (cb[1] + cb[3]) / 2.0)
                pc = ((pb[0] + pb[2]) / 2.0, (pb[1] + pb[3]) / 2.0)
                draw.line([cc, pc], fill=(220, 220, 220), width=1)

    # 4. Save
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_VIS_DIR, f"{clean_id}_full_hierarchy.png")
    img.save(out_path)
    return out_path


def main():
    json_files = sorted([f for f in os.listdir(MATCH_RESULTS_DIR) if f.endswith('_full_hierarchy.json')])
    for jf in json_files:
        clean_id = jf.replace('_full_hierarchy.json', '')
        visualize_full_hierarchy(clean_id)
    print(f"=== ĐÃ VẼ XONG ẢNH VISUALIZED FULL HIERARCHY TẠI: {OUTPUT_VIS_DIR} ===")


if __name__ == "__main__":
    main()
