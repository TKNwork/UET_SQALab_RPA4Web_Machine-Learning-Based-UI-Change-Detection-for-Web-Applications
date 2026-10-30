import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Thư mục gốc project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'yolo_pixelweb_dataset', 'images', 'val')
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'yolo_pixelweb_dataset', 'images', 'val')

MATCH_RESULTS_DIR = os.path.join(BASE_DIR, 'combine', 'small_x_ocr', 'results', 'json')
OCR_RESULTS_DIR = os.path.join(BASE_DIR, 'ocr', 'results')
OUTPUT_VIS_DIR = os.path.join(BASE_DIR, 'combine', 'small_x_ocr', 'results', 'visualized')

# Cấu hình màu sắc & style
COLOR_MAP = {
    'duplicate_ocr': (204, 0, 0),               # Đỏ đậm (#CC0000)
    'nested_in_ocr': (255, 140, 0),             # Cam (#FF8C00)
    'partial_overlap_review': (255, 215, 0),    # Vàng (#FFD700)
    'standalone_pending_yolo_check': (136, 136, 136) # Xám (#888888)
}

STYLE_MAP = {
    'duplicate_ocr': 'solid',
    'nested_in_ocr': 'solid',
    'partial_overlap_review': 'dashed',
    'standalone_pending_yolo_check': 'dotted'
}

ABBR_MAP = {
    'duplicate_ocr': 'dup',
    'nested_in_ocr': 'nested',
    'partial_overlap_review': 'partial',
    'standalone_pending_yolo_check': 'standalone'
}

COLOR_OCR = (0, 255, 0)         # Xanh lá (#00FF00)
COLOR_CONNECT_LINE = (204, 204, 204) # Xám nhạt (#CCCCCC)


def draw_dashed_line(draw, pt1, pt2, fill, width=2, dash_len=6, gap_len=4):
    """Vẽ đường nét đứt/chấm giữa pt1 và pt2"""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist

    curr = 0.0
    while curr < dist:
        end = min(curr + dash_len, dist)
        p_start = (x1 + dx * curr, y1 + dy * curr)
        p_end = (x1 + dx * end, y1 + dy * end)
        draw.line([p_start, p_end], fill=fill, width=width)
        curr += dash_len + gap_len


def draw_styled_rectangle(draw, bbox, color, style='solid', width=2):
    """Vẽ bounding box với kiểu đường vẽ chỉ định (solid, dashed, dotted)"""
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    if style == 'solid':
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    elif style == 'dashed':
        for i in range(4):
            draw_dashed_line(draw, corners[i], corners[(i + 1) % 4], fill=color, width=width, dash_len=6, gap_len=4)
    elif style == 'dotted':
        for i in range(4):
            draw_dashed_line(draw, corners[i], corners[(i + 1) % 4], fill=color, width=width, dash_len=2, gap_len=3)


def draw_legend(draw, font):
    """Vẽ bảng chú thích (Legend) ở góc trên bên trái của ảnh"""
    legend_box = [10, 10, 270, 135]
    draw.rectangle(legend_box, fill=(0, 0, 0), outline=(255, 255, 255), width=1)

    draw.text((15, 14), "Legend (Small YOLO x OCR):", fill=(255, 255, 255), font=font)

    items = [
        ("dup (duplicate_ocr)", COLOR_MAP['duplicate_ocr'], 'solid'),
        ("nested (nested_in_ocr)", COLOR_MAP['nested_in_ocr'], 'solid'),
        ("partial (partial_overlap)", COLOR_MAP['partial_overlap_review'], 'dashed'),
        ("standalone (pending)", COLOR_MAP['standalone_pending_yolo_check'], 'dotted'),
        ("OCR Paragraphs", COLOR_OCR, 'solid')
    ]

    y_offset = 32
    for label, color, style in items:
        # Sample line/box
        sample_box = [15, y_offset + 2, 35, y_offset + 10]
        draw_styled_rectangle(draw, sample_box, color, style=style, width=2)
        draw.text((42, y_offset - 2), label, fill=color if label != "OCR Paragraphs" else COLOR_OCR, font=font)
        y_offset += 20


def visualize_image_match(image_name: str) -> str:
    """
    Vẽ overlay kết quả so khớp cho 1 ảnh và lưu vào combine/small_x_ocr/results/visualized/
    """
    img_id = image_name.replace('.png', '').replace('_small_ocr_match', '')

    # Đường dẫn file
    img_path = os.path.join(VAL_IMAGES_DIR, f"{img_id}.png")
    match_json_path = os.path.join(MATCH_RESULTS_DIR, f"{img_id}_small_ocr_match.json")
    ocr_json_path = os.path.join(OCR_RESULTS_DIR, f"{img_id}.json")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh gốc: {img_path}")
    if not os.path.exists(match_json_path):
        raise FileNotFoundError(f"Không tìm thấy JSON matching: {match_json_path}")
    if not os.path.exists(ocr_json_path):
        raise FileNotFoundError(f"Không tìm thấy JSON OCR gốc: {ocr_json_path}")

    # Load ảnh & JSON
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    with open(ocr_json_path, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)

    with open(match_json_path, 'r', encoding='utf-8') as f:
        match_data = json.load(f)

    ocr_paragraphs = ocr_data.get('paragraphs', [])
    matches = match_data.get('matches', [])

    # 1. Vẽ toàn bộ OCR boxes (nền) màu xanh lá
    for ocr_item in ocr_paragraphs:
        bbox = ocr_item.get('bbox', [0, 0, 0, 0])
        if len(bbox) == 4 and (bbox[2] > bbox[0]) and (bbox[3] > bbox[1]):
            draw.rectangle(bbox, outline=COLOR_OCR, width=1)

    # Helper tìm tâm OCR box theo text
    def find_ocr_center_and_bbox(matched_text: str, small_bbox: list):
        if not matched_text:
            return None
        candidates = [p for p in ocr_paragraphs if p.get('text') == matched_text]
        if not candidates:
            return None
        if len(candidates) == 1:
            ob = candidates[0]['bbox']
            return ((ob[0] + ob[2]) / 2.0, (ob[1] + ob[3]) / 2.0)

        # Nếu có nhiều paragraph trùng text, chọn box gần tâm small nhất
        sx_c = (small_bbox[0] + small_bbox[2]) / 2.0
        sy_c = (small_bbox[1] + small_bbox[3]) / 2.0

        best_center = None
        min_dist = float('inf')
        for cand in candidates:
            ob = cand['bbox']
            cx = (ob[0] + ob[2]) / 2.0
            cy = (ob[1] + ob[3]) / 2.0
            dist = (cx - sx_c) ** 2 + (cy - sy_c) ** 2
            if dist < min_dist:
                min_dist = dist
                best_center = (cx, cy)
        return best_center

    # 2. Vẽ từng Small YOLO box & Đường nối & Label text
    for m in matches:
        bbox = m['bbox']
        case = m['case']
        matched_text = m.get('primary_ocr_match')

        color = COLOR_MAP.get(case, (136, 136, 136))
        style = STYLE_MAP.get(case, 'solid')
        abbr = ABBR_MAP.get(case, case)

        # 2a. Vẽ đường nối nếu có OCR match
        if matched_text:
            ocr_center = find_ocr_center_and_bbox(matched_text, bbox)
            if ocr_center:
                small_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                draw.line([small_center, ocr_center], fill=COLOR_CONNECT_LINE, width=1)

        # 2b. Vẽ Small YOLO box
        draw_styled_rectangle(draw, bbox, color=color, style=style, width=2)

        # 2c. Label text ngay phía trên box
        label_str = abbr
        text_pos = (bbox[0], max(0, bbox[1] - 11))
        draw.text(text_pos, label_str, fill=color, font=font)

    # 3. Vẽ Legend ở góc trên ảnh
    draw_legend(draw, font)

    # 4. Save output
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    output_filename = f"{img_id}_small_ocr_match.png"
    output_path = os.path.join(OUTPUT_VIS_DIR, output_filename)
    img.save(output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Vẽ overlay kết quả so khớp Small YOLO x OCR")
    parser.add_argument("--image", type=str, default=None, help="Mã ảnh (vd: b1_1020 hoặc b1_1020.png)")
    parser.add_argument("--all", action="store_true", help="Chạy vẽ cho tất cả ảnh trong results/json")
    args = parser.parse_args()

    if args.image:
        output_path = visualize_image_match(args.image)
        print(f"-> Đã vẽ trực quan hóa cho ảnh '{args.image}': {output_path}")
    elif args.all:
        json_files = sorted([f for f in os.listdir(MATCH_RESULTS_DIR) if f.endswith('_small_ocr_match.json')])
        print(f"-> Bắt đầu vẽ cho {len(json_files)} ảnh...")
        for jf in json_files:
            img_id = jf.replace('_small_ocr_match.json', '')
            out_p = visualize_image_match(img_id)
            print(f"  + [{img_id}] -> {os.path.basename(out_p)}")
        print(f"=== HOÀN THÀNH VẼ TẤT CẢ ẢNH TẠI: {OUTPUT_VIS_DIR} ===")
    else:
        # Mặc định chạy thử nghiệm ảnh b1_1020
        output_path = visualize_image_match("b1_1020")
        print(f"-> [DEFAULT TEST] Đã vẽ trực quan hóa cho b1_1020: {output_path}")


if __name__ == "__main__":
    main()
