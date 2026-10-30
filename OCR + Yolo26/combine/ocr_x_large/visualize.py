import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from combine._core.schema import load_ocr, load_large

VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'yolo_pixelweb_dataset', 'images', 'val')
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'yolo_pixelweb_dataset', 'images', 'val')

MATCH_RESULTS_DIR = os.path.join(BASE_DIR, 'combine', 'ocr_x_large', 'results', 'json')
OUTPUT_VIS_DIR = os.path.join(BASE_DIR, 'combine', 'ocr_x_large', 'results', 'visualized')

COLOR_MAP = {
    'duplicate': (204, 0, 0),         # Đỏ đậm
    'a_nested_in_b': (255, 140, 0),   # Cam (OCR in Large)
    'b_nested_in_a': (147, 112, 219), # Tím (Large in OCR)
    'partial_overlap': (255, 215, 0), # Vàng
    'standalone': (136, 136, 136)     # Xám
}

COLOR_OCR = (0, 255, 0)        # Xanh lá
COLOR_LARGE = (0, 200, 0)      # Xanh lá đậm


def visualize_ocr_x_large(image_name: str) -> str:
    clean_id = image_name.replace('.png', '').replace('_ocr_large', '')

    img_path = os.path.join(VAL_IMAGES_DIR, f"{clean_id}.png")
    match_json_path = os.path.join(MATCH_RESULTS_DIR, f"{clean_id}_ocr_large.json")

    if not os.path.exists(img_path) or not os.path.exists(match_json_path):
        return ""

    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    ocr_objs = load_ocr(clean_id, BASE_DIR)
    large_objs = load_large(clean_id, BASE_DIR)

    with open(match_json_path, 'r', encoding='utf-8') as f:
        match_data = json.load(f)

    relations = match_data.get('relations', [])

    all_objs = {o['id']: o for o in ocr_objs + large_objs}

    for l_obj in large_objs:
        draw.rectangle(l_obj['bbox'], outline=COLOR_LARGE, width=1)

    for o_obj in ocr_objs:
        draw.rectangle(o_obj['bbox'], outline=COLOR_OCR, width=1)

    for r in relations:
        o_id, l_id, case = r['a_id'], r['b_id'], r['case']
        if o_id in all_objs and l_id in all_objs:
            ob, lb = all_objs[o_id]['bbox'], all_objs[l_id]['bbox']
            oc = ((ob[0] + ob[2]) / 2.0, (ob[1] + ob[3]) / 2.0)
            lc = ((lb[0] + lb[2]) / 2.0, (lb[1] + lb[3]) / 2.0)
            c = COLOR_MAP.get(case, (204, 204, 204))
            draw.line([oc, lc], fill=c, width=2)
            draw.text((oc[0], oc[1] - 10), case, fill=c, font=font)

    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_VIS_DIR, f"{clean_id}_ocr_large.png")
    img.save(out_path)
    return out_path


def main():
    json_files = sorted([f for f in os.listdir(MATCH_RESULTS_DIR) if f.endswith('_ocr_large.json')])
    for jf in json_files:
        clean_id = jf.replace('_ocr_large.json', '')
        visualize_ocr_x_large(clean_id)
    print(f"=== ĐÃ VẼ XONG ẢNH VISUALIZED OCR x LARGE TẠI: {OUTPUT_VIS_DIR} ===")


if __name__ == "__main__":
    main()
