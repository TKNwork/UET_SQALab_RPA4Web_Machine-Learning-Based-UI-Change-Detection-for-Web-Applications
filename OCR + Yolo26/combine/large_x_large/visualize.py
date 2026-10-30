import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from combine._core.schema import load_large

VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'yolo_pixelweb_dataset', 'images', 'val')
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'yolo_pixelweb_dataset', 'images', 'val')

MATCH_RESULTS_DIR = os.path.join(BASE_DIR, 'combine', 'large_x_large', 'results', 'json')
OUTPUT_VIS_DIR = os.path.join(BASE_DIR, 'combine', 'large_x_large', 'results', 'visualized')

COLOR_MAP = {
    'duplicate': (204, 0, 0),         # Đỏ đậm
    'a_nested_in_b': (255, 140, 0),   # Cam (Large A in Large B)
    'b_nested_in_a': (147, 112, 219), # Tím (Large B in Large A)
    'partial_overlap': (255, 215, 0), # Vàng
    'standalone': (136, 136, 136)     # Xám
}

COLOR_LARGE = (0, 200, 0)      # Xanh lá


def visualize_large_x_large(image_name: str) -> str:
    clean_id = image_name.replace('.png', '').replace('_large_large', '')

    img_path = os.path.join(VAL_IMAGES_DIR, f"{clean_id}.png")
    match_json_path = os.path.join(MATCH_RESULTS_DIR, f"{clean_id}_large_large.json")

    if not os.path.exists(img_path) or not os.path.exists(match_json_path):
        return ""

    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    large_objs = load_large(clean_id, BASE_DIR)

    with open(match_json_path, 'r', encoding='utf-8') as f:
        match_data = json.load(f)

    relations = match_data.get('relations', [])

    all_objs = {o['id']: o for o in large_objs}

    for l_obj in large_objs:
        draw.rectangle(l_obj['bbox'], outline=COLOR_LARGE, width=1)

    for r in relations:
        la_id, lb_id, case = r['a_id'], r['b_id'], r['case']
        if la_id in all_objs and lb_id in all_objs:
            lab, lbb = all_objs[la_id]['bbox'], all_objs[lb_id]['bbox']
            lac = ((lab[0] + lab[2]) / 2.0, (lab[1] + lab[3]) / 2.0)
            lbc = ((lbb[0] + lbb[2]) / 2.0, (lbb[1] + lbb[3]) / 2.0)
            c = COLOR_MAP.get(case, (204, 204, 204))
            draw.line([lac, lbc], fill=c, width=2)
            draw.text((lac[0], lac[1] - 10), case, fill=c, font=font)

    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_VIS_DIR, f"{clean_id}_large_large.png")
    img.save(out_path)
    return out_path


def main():
    json_files = sorted([f for f in os.listdir(MATCH_RESULTS_DIR) if f.endswith('_large_large.json')])
    for jf in json_files:
        clean_id = jf.replace('_large_large.json', '')
        visualize_large_x_large(clean_id)
    print(f"=== ĐÃ VẼ XONG ẢNH VISUALIZED LARGE x LARGE TẠI: {OUTPUT_VIS_DIR} ===")


if __name__ == "__main__":
    main()
