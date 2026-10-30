import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from combine._core.schema import load_small, load_medium, load_large

VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'yolo_pixelweb_dataset', 'images', 'val')
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'yolo_pixelweb_dataset', 'images', 'val')

MATCH_RESULTS_DIR = os.path.join(BASE_DIR, 'combine', 'small_x_yolo', 'results', 'json')
OUTPUT_VIS_DIR = os.path.join(BASE_DIR, 'combine', 'small_x_yolo', 'results', 'visualized')

COLOR_MAP = {
    'duplicate': (204, 0, 0),         # Đỏ đậm
    'a_nested_in_b': (255, 140, 0),   # Cam (Small in Medium/Large)
    'b_nested_in_a': (147, 112, 219), # Tím
    'partial_overlap': (255, 215, 0), # Vàng
    'standalone': (136, 136, 136)     # Xám
}

COLOR_SMALL = (255, 0, 0)      # Đỏ
COLOR_MEDIUM = (0, 100, 255)   # Xanh dương
COLOR_LARGE = (0, 200, 0)      # Xanh lá


def visualize_small_x_yolo(image_name: str) -> str:
    clean_id = image_name.replace('.png', '').replace('_small_yolo', '')

    img_path = os.path.join(VAL_IMAGES_DIR, f"{clean_id}.png")
    match_json_path = os.path.join(MATCH_RESULTS_DIR, f"{clean_id}_small_yolo.json")

    if not os.path.exists(img_path) or not os.path.exists(match_json_path):
        return ""

    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    small_objs = load_small(clean_id, BASE_DIR)
    medium_objs = load_medium(clean_id, BASE_DIR)
    large_objs = load_large(clean_id, BASE_DIR)

    with open(match_json_path, 'r', encoding='utf-8') as f:
        match_data = json.load(f)

    rels_sm = match_data.get('relations_small_medium', [])
    rels_sl = match_data.get('relations_small_large', [])

    all_objs = {o['id']: o for o in small_objs + medium_objs + large_objs}

    # 1. Vẽ Large Boxes (nền)
    for l_obj in large_objs:
        draw.rectangle(l_obj['bbox'], outline=COLOR_LARGE, width=1)

    # 2. Vẽ Medium Boxes (nền)
    for m_obj in medium_objs:
        draw.rectangle(m_obj['bbox'], outline=COLOR_MEDIUM, width=1)

    # 3. Vẽ Small Boxes
    for s_obj in small_objs:
        draw.rectangle(s_obj['bbox'], outline=COLOR_SMALL, width=1)

    # 4. Vẽ đường nối cho Small x Medium
    for r in rels_sm:
        s_id, m_id, case = r['a_id'], r['b_id'], r['case']
        if s_id in all_objs and m_id in all_objs:
            sb, mb = all_objs[s_id]['bbox'], all_objs[m_id]['bbox']
            sc = ((sb[0] + sb[2]) / 2.0, (sb[1] + sb[3]) / 2.0)
            mc = ((mb[0] + mb[2]) / 2.0, (mb[1] + mb[3]) / 2.0)
            c = COLOR_MAP.get(case, (204, 204, 204))
            draw.line([sc, mc], fill=c, width=2)

    # 5. Vẽ đường nối cho Small x Large
    for r in rels_sl:
        s_id, l_id, case = r['a_id'], r['b_id'], r['case']
        if s_id in all_objs and l_id in all_objs:
            sb, lb = all_objs[s_id]['bbox'], all_objs[l_id]['bbox']
            sc = ((sb[0] + sb[2]) / 2.0, (sb[1] + sb[3]) / 2.0)
            lc = ((lb[0] + lb[2]) / 2.0, (lb[1] + lb[3]) / 2.0)
            c = COLOR_MAP.get(case, (204, 204, 204))
            draw.line([sc, lc], fill=c, width=1)

    # Save
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_VIS_DIR, f"{clean_id}_small_yolo.png")
    img.save(out_path)
    return out_path


def main():
    json_files = sorted([f for f in os.listdir(MATCH_RESULTS_DIR) if f.endswith('_small_yolo.json')])
    for jf in json_files:
        clean_id = jf.replace('_small_yolo.json', '')
        visualize_small_x_yolo(clean_id)
    print(f"=== ĐÃ VẼ XONG ẢNH VISUALIZED SMALL x YOLO TẠI: {OUTPUT_VIS_DIR} ===")


if __name__ == "__main__":
    main()
