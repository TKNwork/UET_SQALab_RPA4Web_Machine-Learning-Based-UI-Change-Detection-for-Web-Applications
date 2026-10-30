import os
import sys
import json
import time
from PIL import Image, ImageDraw, ImageFont

# Xác định thư mục gốc của project (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cấu hình ngưỡng confidence theo từng size_category mới:
# large >= 0.85, med >= 0.50, small >= 0.10
SIZE_THRESHOLDS = {
    'small': 0.10,
    'medium': 0.50,
    'large': 0.85
}

# Màu sắc phân biệt theo kích thước (RGB)
SIZE_COLORS = {
    'small': (255, 0, 0),      # Đỏ (Red)
    'medium': (0, 100, 255),   # Xanh Dương (Blue)
    'large': (0, 200, 0)       # Xanh Lá (Green)
}

# Hỗ trợ cả thư mục gốc ./yolo_pixelweb_dataset và ./data/yolo_pixelweb_dataset
VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'yolo_pixelweb_dataset', 'images', 'val')
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'yolo_pixelweb_dataset', 'images', 'val')

ANALYSIS_DIR = os.path.join(BASE_DIR, 'yolo', 'analysis')
STATS_JSON_PATH = os.path.join(ANALYSIS_DIR, 'size_confidence_stats.json')

# Thư mục visualized mới chứa các ảnh lọc theo ngưỡng v2 (large>=0.85, med>=0.5, small>=0.1)
FILTERED_VIS_DIR = os.path.join(ANALYSIS_DIR, 'visualized_filtered_v2')
VIS_SMALL_DIR = os.path.join(FILTERED_VIS_DIR, 'small')
VIS_MEDIUM_DIR = os.path.join(FILTERED_VIS_DIR, 'medium')
VIS_LARGE_DIR = os.path.join(FILTERED_VIS_DIR, 'large')
VIS_COMBINED_DIR = os.path.join(FILTERED_VIS_DIR, 'combined')
FILTERED_SUMMARY_JSON = os.path.join(FILTERED_VIS_DIR, 'filtered_summary.json')


def get_size_category(area):
    """
    Phân loại kích thước bbox theo chuẩn COCO:
    - small : area < 32^2 (1024 px^2)
    - medium: 32^2 <= area < 96^2 (1024 <= area < 9216 px^2)
    - large : area >= 96^2 (9216 px^2)
    """
    if area < 1024:
        return 'small'
    elif area < 9216:
        return 'medium'
    else:
        return 'large'


def draw_filtered_image(image_path, detections, target_size, output_image_path):
    """
    Vẽ ảnh lọc theo target_size ('small', 'medium', 'large', hoặc 'combined')
    chỉ giữ lại các phần tử thỏa mãn ngưỡng confidence tương ứng:
    - small  >= 0.10
    - medium >= 0.50
    - large  >= 0.85
    """
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Lọc danh sách detections
    filtered_dets = []
    for det in detections:
        cat = det.get('size_category')
        conf = det.get('confidence', 0.0)
        thresh = SIZE_THRESHOLDS.get(cat, 0.0)
        
        if conf >= thresh:
            if target_size == 'combined' or cat == target_size:
                filtered_dets.append(det)

    # Vẽ từng bbox
    for det in filtered_dets:
        bbox = det.get('bbox', [])
        conf = det.get('confidence', 0.0)
        cat = det.get('size_category', 'small')
        box_color = SIZE_COLORS.get(cat, (255, 0, 0))

        if len(bbox) == 4:
            x_min, y_min, x_max, y_max = bbox
            draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=2)

            label = f"{cat} {conf:.2f}"
            text_pos = (x_min, max(0, y_min - 12))

            text_bbox = draw.textbbox(text_pos, label, font=font) if hasattr(draw, 'textbbox') else None
            if text_bbox:
                draw.rectangle(text_bbox, fill=(0, 0, 0))
            draw.text(text_pos, label, fill=(255, 255, 255), font=font)

    img.save(output_image_path)
    return len(filtered_dets)


def load_or_run_detections():
    """Lấy dữ liệu detections từ file stats JSON sẵn có hoặc chạy trực tiếp mô hình"""
    if os.path.exists(STATS_JSON_PATH):
        print(f"Đang đọc dữ liệu detections sẵn có từ: {os.path.relpath(STATS_JSON_PATH, BASE_DIR)}")
        with open(STATS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('per_image', [])
    else:
        print(f"Không tìm thấy {STATS_JSON_PATH}, đang tiến hành nạp mô hình YOLO...")
        MODEL_PATH = os.path.join(BASE_DIR, 'yolo', 'models', 'yl26best.pt')
        if not os.path.exists(MODEL_PATH):
            MODEL_PATH = os.path.join(BASE_DIR, 'yl26best.pt')
            
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)

        files = sorted([
            f for f in os.listdir(VAL_IMAGES_DIR)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'))
        ])[:30]

        per_image = []
        for img_name in files:
            img_path = os.path.join(VAL_IMAGES_DIR, img_name)
            # Chạy inference với conf=0.1 để bắt toàn bộ các detection small >= 0.1
            results = model.predict(source=img_path, conf=0.10, verbose=False)
            boxes = results[0].boxes

            img_dets = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()
                    x_min, y_min, x_max, y_max = xyxy
                    area = round(max(0, x_max - x_min) * max(0, y_max - y_min), 2)
                    size_cat = get_size_category(area)
                    confidence = round(float(box.conf[0]), 4)
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, str(cls_id)) if hasattr(model, 'names') else str(cls_id)

                    img_dets.append({
                        "class": cls_name,
                        "confidence": confidence,
                        "bbox": [round(coord, 2) for coord in xyxy],
                        "area": area,
                        "size_category": size_cat
                    })
            per_image.append({"image": img_name, "detections": img_dets})

        return per_image


def main():
    print("=== BẮT ĐẦU VẼ VÀ LỌC ẢNH TRỰC QUAN THEO NGƯỠNG V2 ===")
    print(f"Đường dẫn ảnh dataset: {os.path.relpath(VAL_IMAGES_DIR, BASE_DIR)}")
    print("Ngưỡng lọc mới được áp dụng:")
    print(f" - Large  (>= 96²)  : Confidence >= {SIZE_THRESHOLDS['large']:.2f}")
    print(f" - Medium (32²-96²) : Confidence >= {SIZE_THRESHOLDS['medium']:.2f}")
    print(f" - Small  (< 32²)   : Confidence >= {SIZE_THRESHOLDS['small']:.2f}")
    print("-" * 60)

    # 1. Tạo các thư mục output
    os.makedirs(VIS_SMALL_DIR, exist_ok=True)
    os.makedirs(VIS_MEDIUM_DIR, exist_ok=True)
    os.makedirs(VIS_LARGE_DIR, exist_ok=True)
    os.makedirs(VIS_COMBINED_DIR, exist_ok=True)

    # 2. Đọc dữ liệu detections
    per_image_data = load_or_run_detections()
    print(f"Đã nạp dữ liệu cho {len(per_image_data)} ảnh.")
    print("-" * 60)

    summary_per_image = []

    total_counts = {'small': 0, 'medium': 0, 'large': 0, 'combined': 0}
    orig_counts = {'small': 0, 'medium': 0, 'large': 0, 'total': 0}

    # 3. Tiến hành vẽ và xuất ảnh
    start_time = time.time()
    for idx, item in enumerate(per_image_data, 1):
        img_name = item.get("image")
        detections = item.get("detections", [])

        img_path = os.path.join(VAL_IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"⚠️ [BỎ QUA] Không tìm thấy ảnh gốc: {img_path}")
            continue

        base_name, _ = os.path.splitext(img_name)

        # Thống kê nguyên bản
        for d in detections:
            cat = d.get('size_category')
            if cat in orig_counts:
                orig_counts[cat] += 1
            orig_counts['total'] += 1

        # Đường dẫn ảnh đầu ra
        small_out = os.path.join(VIS_SMALL_DIR, f"{base_name}_small.png")
        med_out = os.path.join(VIS_MEDIUM_DIR, f"{base_name}_medium.png")
        large_out = os.path.join(VIS_LARGE_DIR, f"{base_name}_large.png")
        combined_out = os.path.join(VIS_COMBINED_DIR, f"{base_name}_combined.png")

        # Vẽ từng loại ảnh
        cnt_s = draw_filtered_image(img_path, detections, 'small', small_out)
        cnt_m = draw_filtered_image(img_path, detections, 'medium', med_out)
        cnt_l = draw_filtered_image(img_path, detections, 'large', large_out)
        cnt_c = draw_filtered_image(img_path, detections, 'combined', combined_out)

        total_counts['small'] += cnt_s
        total_counts['medium'] += cnt_m
        total_counts['large'] += cnt_l
        total_counts['combined'] += cnt_c

        summary_per_image.append({
            "image": img_name,
            "original_detections_count": len(detections),
            "filtered_small_count": cnt_s,
            "filtered_medium_count": cnt_m,
            "filtered_large_count": cnt_l,
            "filtered_combined_count": cnt_c
        })

        print(f"[{idx}/{len(per_image_data)}] {img_name} -> Gộp: {cnt_c} | Small: {cnt_s} | Medium: {cnt_m} | Large: {cnt_l}")

    elapsed = time.time() - start_time

    # 4. Lưu JSON summary
    summary_data = {
        "thresholds": SIZE_THRESHOLDS,
        "overall_summary": {
            "total_images": len(per_image_data),
            "original_total_detections": orig_counts['total'],
            "filtered_total_detections": total_counts['combined'],
            "small": {"original": orig_counts['small'], "filtered": total_counts['small'], "threshold": SIZE_THRESHOLDS['small']},
            "medium": {"original": orig_counts['medium'], "filtered": total_counts['medium'], "threshold": SIZE_THRESHOLDS['medium']},
            "large": {"original": orig_counts['large'], "filtered": total_counts['large'], "threshold": SIZE_THRESHOLDS['large']}
        },
        "per_image_summary": summary_per_image
    }

    with open(FILTERED_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    # 5. In bảng tổng kết
    print("\n" + "=" * 60)
    print("TỔNG KẾT BÁO CÁO LỌC ẢNH THEO THRESHOLD CUSTOM V2")
    print("=" * 60)
    print(f"  - Thư mục ảnh lọc V2 : {os.path.relpath(FILTERED_VIS_DIR, BASE_DIR)}/")
    print(f"    * Small  (conf >= {SIZE_THRESHOLDS['small']:.2f})  : {total_counts['small']}/{orig_counts['small']} detections -> {os.path.relpath(VIS_SMALL_DIR, BASE_DIR)}/")
    print(f"    * Medium (conf >= {SIZE_THRESHOLDS['medium']:.2f}) : {total_counts['medium']}/{orig_counts['medium']} detections -> {os.path.relpath(VIS_MEDIUM_DIR, BASE_DIR)}/")
    print(f"    * Large  (conf >= {SIZE_THRESHOLDS['large']:.2f})  : {total_counts['large']}/{orig_counts['large']} detections -> {os.path.relpath(VIS_LARGE_DIR, BASE_DIR)}/")
    print(f"    * Combined (Tất cả)     : {total_counts['combined']}/{orig_counts['total']} detections -> {os.path.relpath(VIS_COMBINED_DIR, BASE_DIR)}/")
    print(f"  - File JSON Summary : {os.path.relpath(FILTERED_SUMMARY_JSON, BASE_DIR)}")
    print(f"  - Thời gian xử lý   : {elapsed:.2f} giây")
    print("=" * 60)


if __name__ == "__main__":
    main()
