import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from combine._core.schema import load_small, load_cv

VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'yolo_pixelweb_dataset', 'images', 'val')
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'yolo_pixelweb_dataset', 'images', 'val')

MATCH_RESULTS_DIR = os.path.join(BASE_DIR, 'combine', 'small_x_cv', 'results', 'json')
OUTPUT_VIS_DIR = os.path.join(BASE_DIR, 'combine', 'small_x_cv', 'results', 'visualized')

COLOR_MAP = {
    'duplicate': (204, 0, 0),         # Đỏ đậm
    'a_nested_in_b': (255, 140, 0),   # Cam (Small in CV)
    'b_nested_in_a': (147, 112, 219), # Tím (CV in Small)
    'partial_overlap': (255, 215, 0), # Vàng
    'standalone': (136, 136, 136)     # Xám
}

COLOR_SMALL = (255, 0, 0)   # Đỏ
COLOR_CV = (0, 100, 255)    # Xanh dương
COLOR_LINE = (204, 204, 204) # Xám nhạt


def visualize_small_x_cv(image_name: str) -> str:
    clean_id = image_name.replace('.png', '').replace('_small_cv', '')

    img_path = os.path.join(VAL_IMAGES_DIR, f"{clean_id}.png")
    match_json_path = os.path.join(MATCH_RESULTS_DIR, f"{clean_id}_small_cv.json")

    if not os.path.exists(img_path) or not os.path.exists(match_json_path):
        return ""

    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    small_objs = load_small(clean_id, BASE_DIR)
    cv_objs = load_cv(clean_id, BASE_DIR)

    with open(match_json_path, 'r', encoding='utf-8') as f:
        match_data = json.load(f)

    relations = match_data.get('relations', [])

    # Map ID -> obj
    small_dict = {o['id']: o for o in small_objs}
    cv_dict = {o['id']: o for o in cv_objs}

    # 1. Vẽ CV Boxes (nền)
    for c_obj in cv_objs:
        draw.rectangle(c_obj['bbox'], outline=COLOR_CV, width=1)

    # 2. Vẽ Small Boxes
    for s_obj in small_objs:
        draw.rectangle(s_obj['bbox'], outline=COLOR_SMALL, width=1)

    # 3. Vẽ đường nối & label quan hệ
    for r in relations:
        s_id = r['a_id']
        c_id = r['b_id']
        case = r['case']

        if s_id in small_dict and c_id in cv_dict:
            sb = small_dict[s_id]['bbox']
            cb = cv_dict[c_id]['bbox']

            sc = ((sb[0] + sb[2]) / 2.0, (sb[1] + sb[3]) / 2.0)
            cc = ((cb[0] + cb[2]) / 2.0, (cb[1] + cb[3]) / 2.0)

            line_color = COLOR_MAP.get(case, COLOR_LINE)
            draw.line([sc, cc], fill=line_color, width=2)
            draw.text((sc[0], sc[1] - 10), case, fill=line_color, font=font)

    # 4. Save
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_VIS_DIR, f"{clean_id}_small_cv.png")
    img.save(out_path)
    return out_path


def main():
    json_files = sorted([f for f in os.listdir(MATCH_RESULTS_DIR) if f.endswith('_small_cv.json')])
    for jf in json_files:
        clean_id = jf.replace('_small_cv.json', '')
        visualize_small_x_cv(clean_id)
    print(f"=== ĐÃ VẼ XONG ẢNH VISUALIZED SMALL x CV TẠI: {OUTPUT_VIS_DIR} ===")


if __name__ == "__main__":
    main()
