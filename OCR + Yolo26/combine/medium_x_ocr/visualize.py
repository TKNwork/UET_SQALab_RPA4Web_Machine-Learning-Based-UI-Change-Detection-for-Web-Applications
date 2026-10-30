import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from combine._core.schema import load_medium, load_ocr

VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'yolo_pixelweb_dataset', 'images', 'val')
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'yolo_pixelweb_dataset', 'images', 'val')

MATCH_RESULTS_DIR = os.path.join(BASE_DIR, 'combine', 'medium_x_ocr', 'results', 'json')
OUTPUT_VIS_DIR = os.path.join(BASE_DIR, 'combine', 'medium_x_ocr', 'results', 'visualized')

COLOR_MAP = {
    'duplicate': (204, 0, 0),         # Đỏ đậm
    'a_nested_in_b': (255, 140, 0),   # Cam (Medium in OCR)
    'b_nested_in_a': (0, 100, 255),   # Xanh dương (OCR in Medium)
    'partial_overlap': (255, 215, 0), # Vàng
    'standalone': (136, 136, 136)     # Xám
}

COLOR_MEDIUM = (0, 100, 255)   # Xanh dương
COLOR_OCR = (0, 255, 0)        # Xanh lá


def visualize_medium_x_ocr(image_name: str) -> str:
    clean_id = image_name.replace('.png', '').replace('_medium_ocr', '')

    img_path = os.path.join(VAL_IMAGES_DIR, f"{clean_id}.png")
    match_json_path = os.path.join(MATCH_RESULTS_DIR, f"{clean_id}_medium_ocr.json")

    if not os.path.exists(img_path) or not os.path.exists(match_json_path):
        return ""

    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    medium_objs = load_medium(clean_id, BASE_DIR)
    ocr_objs = load_ocr(clean_id, BASE_DIR)

    with open(match_json_path, 'r', encoding='utf-8') as f:
        match_data = json.load(f)

    relations = match_data.get('relations', [])

    all_objs = {o['id']: o for o in medium_objs + ocr_objs}

    # 1. Vẽ OCR (nền)
    for o_obj in ocr_objs:
        draw.rectangle(o_obj['bbox'], outline=COLOR_OCR, width=1)

    # 2. Vẽ Medium
    for m_obj in medium_objs:
        draw.rectangle(m_obj['bbox'], outline=COLOR_MEDIUM, width=1)

    # 3. Vẽ đường nối & case
    for r in relations:
        m_id, o_id, case = r['a_id'], r['b_id'], r['case']
        if m_id in all_objs and o_id in all_objs:
            mb, ob = all_objs[m_id]['bbox'], all_objs[o_id]['bbox']
            mc = ((mb[0] + mb[2]) / 2.0, (mb[1] + mb[3]) / 2.0)
            oc = ((ob[0] + ob[2]) / 2.0, (ob[1] + ob[3]) / 2.0)
            c = COLOR_MAP.get(case, (204, 204, 204))
            draw.line([mc, oc], fill=c, width=2)
            draw.text((mc[0], mc[1] - 10), case, fill=c, font=font)

    # Save
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_VIS_DIR, f"{clean_id}_medium_ocr.png")
    img.save(out_path)
    return out_path


def main():
    json_files = sorted([f for f in os.listdir(MATCH_RESULTS_DIR) if f.endswith('_medium_ocr.json')])
    for jf in json_files:
        clean_id = jf.replace('_medium_ocr.json', '')
        visualize_medium_x_ocr(clean_id)
    print(f"=== ĐÃ VẼ XONG ẢNH VISUALIZED MEDIUM x OCR TẠI: {OUTPUT_VIS_DIR} ===")


if __name__ == "__main__":
    main()
