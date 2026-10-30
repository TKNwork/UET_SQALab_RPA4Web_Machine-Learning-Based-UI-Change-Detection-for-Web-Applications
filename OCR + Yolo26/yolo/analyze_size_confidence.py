import os
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Xác định thư mục gốc của project (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, 'yolo', 'models', 'yl26best.pt')
VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'yolo_pixelweb_dataset', 'images', 'val')
ANALYSIS_DIR = os.path.join(BASE_DIR, 'yolo', 'analysis')
JSON_OUTPUT_PATH = os.path.join(ANALYSIS_DIR, 'size_confidence_stats.json')
BOXPLOT_OUTPUT_PATH = os.path.join(ANALYSIS_DIR, 'confidence_by_size_boxplot.png')

# Các thư mục lưu ảnh visualized chia theo kích thước
VIS_DIR = os.path.join(ANALYSIS_DIR, 'visualized')
VIS_SMALL_DIR = os.path.join(VIS_DIR, 'small')
VIS_MEDIUM_DIR = os.path.join(VIS_DIR, 'medium')
VIS_LARGE_DIR = os.path.join(VIS_DIR, 'large')


def detect_device():
    """Tự động kiểm tra và trả về device (cuda hoặc cpu)"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"-> Đã phát hiện GPU CUDA: {device_name}")
            return 'cuda'
        else:
            print("-> Không phát hiện GPU CUDA. Sử dụng fallback: CPU")
            return 'cpu'
    except Exception as e:
        print(f"-> Không thể kiểm tra PyTorch CUDA ({e}). Fallback: CPU")
        return 'cpu'


def get_first_n_images(images_dir, n=30):
    if not os.path.exists(images_dir):
        print(f"[LỖI] Thư mục ảnh '{images_dir}' không tồn tại.")
        sys.exit(1)
        
    files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'))
    ])
    
    return files[:n]


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


def draw_size_filtered_image(image_path, detections, target_size, output_image_path):
    """
    Lọc và vẽ chỉ những bbox thuộc target_size (small, medium, hoặc large)
    - small : Màu Đỏ (255, 0, 0)
    - medium: Màu Xanh Dương (0, 100, 255)
    - large : Màu Xanh Lá (0, 200, 0)
    """
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    size_colors = {
        'small': (255, 0, 0),      # Đỏ
        'medium': (0, 100, 255),   # Xanh dương
        'large': (0, 200, 0)       # Xanh lá
    }
    box_color = size_colors.get(target_size, (255, 0, 0))

    filtered_dets = [d for d in detections if d.get('size_category') == target_size]
    
    for det in filtered_dets:
        bbox = det.get('bbox', [])
        conf = det.get('confidence', 0.0)
        
        if len(bbox) == 4:
            x_min, y_min, x_max, y_max = bbox
            draw.rectangle([x_min, y_min, x_max, y_max], outline=box_color, width=2)
            
            label = f"{target_size} {conf:.2f}"
            text_pos = (x_min, max(0, y_min - 12))
            
            text_bbox = draw.textbbox(text_pos, label, font=font) if hasattr(draw, 'textbbox') else None
            if text_bbox:
                draw.rectangle(text_bbox, fill=(0, 0, 0))
            draw.text(text_pos, label, fill=(255, 255, 255), font=font)

    img.save(output_image_path)


def calc_stats(confs):
    """Tính các giá trị thống kê cho 1 mảng confidence"""
    if not confs:
        return {
            "count": 0,
            "mean_conf": 0.0,
            "min_conf": 0.0,
            "max_conf": 0.0,
            "std_conf": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0
        }
    
    arr = np.array(confs)
    return {
        "count": len(confs),
        "mean_conf": round(float(np.mean(arr)), 4),
        "min_conf": round(float(np.min(arr)), 4),
        "max_conf": round(float(np.max(arr)), 4),
        "std_conf": round(float(np.std(arr)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4)
    }


def print_table(title, headers, rows):
    """In bảng formatted đẹp mắt ra console"""
    print(f"\n=== {title} ===")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
            
    header_str = " | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
    sep_str = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_str)
    print(sep_str)
    
    for row in rows:
        row_str = " | ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(row)))
        print(row_str)


def draw_boxplot(confs_by_size, output_path):
    """Vẽ và lưu biểu đồ Boxplot so sánh phân bố confidence theo 3 nhóm kích thước"""
    categories = ['small', 'medium', 'large']
    data = [confs_by_size[cat] for cat in categories]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    filtered_data = [d if len(d) > 0 else [0] for d in data]
    
    bp = ax.boxplot(filtered_data, tick_labels=['small (<32²)', 'medium (32²-96²)', 'large (≥96²)'], patch_artist=True)
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        
    ax.set_title("Phân bố Confidence theo Kích thước Đối tượng (COCO Standard)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Size Category", fontsize=11, labelpad=10)
    ax.set_ylabel("Confidence Score", fontsize=11, labelpad=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    for i, cat in enumerate(categories, 1):
        count = len(confs_by_size[cat])
        ax.text(i, 0.03, f"n = {count}", horizontalalignment='center', size=10, weight='bold', color='black')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    print("=== PHÂN TÍCH YOLO CONFIDENCE THEO KÍCH THƯỚC & VẼ ẢNH TRỰC QUAN ===")
    
    # 1. Detect device
    device = detect_device()
    
    # 2. Check model
    if not os.path.exists(MODEL_PATH):
        print(f"[LỖI] File weights YOLO '{MODEL_PATH}' không tồn tại.")
        sys.exit(1)
        
    # 3. Load YOLO model
    print(f"Đang nạp mô hình YOLO từ: {MODEL_PATH}")
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[LỖI CRITICAL] Không thể nạp mô hình YOLO: {e}")
        sys.exit(1)
        
    # 4. Lấy 30 ảnh đầu tiên
    images = get_first_n_images(VAL_IMAGES_DIR, n=30)
    print(f"Đã tìm thấy {len(images)} ảnh đầu tiên trong '{VAL_IMAGES_DIR}' để phân tích.")
    print("-" * 60)
    
    # Tạo các thư mục lưu kết quả ảnh vẽ
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(VIS_SMALL_DIR, exist_ok=True)
    os.makedirs(VIS_MEDIUM_DIR, exist_ok=True)
    os.makedirs(VIS_LARGE_DIR, exist_ok=True)
    
    per_image_results = []
    all_detections_flat = []
    
    success_count = 0
    fail_count = 0
    
    total_start_time = time.time()
    
    # 5. Chạy inference & phân loại & vẽ ảnh trực quan
    for idx, img_name in enumerate(images, 1):
        img_path = os.path.join(VAL_IMAGES_DIR, img_name)
        base_name, _ = os.path.splitext(img_name)
        
        try:
            results = model.predict(source=img_path, device=device, verbose=False)
            result = results[0]
            boxes = result.boxes
            
            img_dets = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].tolist()  # [x_min, y_min, x_max, y_max]
                    x_min, y_min, x_max, y_max = xyxy
                    width = max(0, x_max - x_min)
                    height = max(0, y_max - y_min)
                    area = round(width * height, 2)
                    
                    size_cat = get_size_category(area)
                    confidence = round(float(box.conf[0]), 4)
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, str(cls_id)) if hasattr(model, 'names') else str(cls_id)
                    
                    det_item = {
                        "class": cls_name,
                        "confidence": confidence,
                        "bbox": [round(coord, 2) for coord in xyxy],
                        "area": area,
                        "size_category": size_cat
                    }
                    img_dets.append(det_item)
                    all_detections_flat.append(det_item)
                    
            per_image_results.append({
                "image": img_name,
                "detections": img_dets
            })
            
            # Vẽ riêng 3 ảnh đại diện cho 3 nhóm size: small, medium, large
            small_img_path = os.path.join(VIS_SMALL_DIR, f"{base_name}_small.png")
            med_img_path = os.path.join(VIS_MEDIUM_DIR, f"{base_name}_medium.png")
            large_img_path = os.path.join(VIS_LARGE_DIR, f"{base_name}_large.png")
            
            draw_size_filtered_image(img_path, img_dets, 'small', small_img_path)
            draw_size_filtered_image(img_path, img_dets, 'medium', med_img_path)
            draw_size_filtered_image(img_path, img_dets, 'large', large_img_path)
            
            success_count += 1
            
            # Đếm số lượng theo size cho ảnh này
            s_cnt = sum(1 for d in img_dets if d['size_category'] == 'small')
            m_cnt = sum(1 for d in img_dets if d['size_category'] == 'medium')
            l_cnt = sum(1 for d in img_dets if d['size_category'] == 'large')
            
            print(f"[{idx}/{len(images)}] {img_name} -> Total: {len(img_dets)} (Small: {s_cnt}, Medium: {m_cnt}, Large: {l_cnt})")
            
        except Exception as e:
            fail_count += 1
            print(f"[CẢNH BÁO] Không thể xử lý ảnh '{img_name}': {e}")
            per_image_results.append({
                "image": img_name,
                "error": str(e),
                "detections": []
            })
            
    print("-" * 60)
    
    # 6. Thống kê theo size category & class_name
    grouped_by_size_class = {}
    grouped_by_size = {"small": [], "medium": [], "large": []}
    
    for det in all_detections_flat:
        cat = det["size_category"]
        cls = det["class"]
        conf = det["confidence"]
        
        key = (cat, cls)
        if key not in grouped_by_size_class:
            grouped_by_size_class[key] = []
        grouped_by_size_class[key].append(conf)
        
        grouped_by_size[cat].append(conf)
        
    by_size_and_class_stats = []
    for (cat, cls), confs in sorted(grouped_by_size_class.items()):
        st = calc_stats(confs)
        st["size_category"] = cat
        st["class"] = cls
        by_size_and_class_stats.append(st)
        
    by_size_only_stats = []
    for cat in ['small', 'medium', 'large']:
        confs = grouped_by_size[cat]
        st = calc_stats(confs)
        st["size_category"] = cat
        by_size_only_stats.append(st)
        
    # 7. In bảng ra Console
    headers_1 = ["Size", "Class", "Count", "Mean", "Min", "Max", "Std", "P25", "P50", "P75"]
    rows_1 = [
        [s["size_category"], s["class"], s["count"], s["mean_conf"], s["min_conf"], s["max_conf"], s["std_conf"], s["p25"], s["p50"], s["p75"]]
        for s in by_size_and_class_stats
    ]
    print_table("THỐNG KÊ CHI TIẾT THEO KÍCH THƯỚC & CLASS", headers_1, rows_1)
    
    headers_2 = ["Size Category", "Count", "Mean Conf", "Min Conf", "Max Conf", "Std Conf", "P25", "P50", "P75"]
    rows_2 = [
        [s["size_category"], s["count"], s["mean_conf"], s["min_conf"], s["max_conf"], s["std_conf"], s["p25"], s["p50"], s["p75"]]
        for s in by_size_only_stats
    ]
    print_table("THỐNG KÊ TỔNG HỢP THEO KÍCH THƯỚC (COCO STANDARD)", headers_2, rows_2)
    
    # 8. Lưu kết quả JSON
    output_json = {
        "per_image": per_image_results,
        "by_size_and_class": by_size_and_class_stats,
        "by_size_only": by_size_only_stats
    }
    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
        
    # 9. Vẽ và lưu Boxplot
    draw_boxplot(grouped_by_size, BOXPLOT_OUTPUT_PATH)
    
    total_elapsed = time.time() - total_start_time
    
    # 10. Tổng kết
    print("\n" + "=" * 60)
    print("TỔNG KẾT BÁO CÁO PHÂN TÍCH KÍCH THƯỚC & VẼ ẢNH TRỰC QUAN")
    print("=" * 60)
    print(f"  - Số ảnh thành công     : {success_count}/{len(images)}")
    print(f"  - Số ảnh thất bại       : {fail_count}/{len(images)}")
    print(f"  - Tổng số detections    : {len(all_detections_flat)}")
    print(f"    * Small  (< 32²)      : {len(grouped_by_size['small'])}")
    print(f"    * Medium (32²-96²)    : {len(grouped_by_size['medium'])}")
    print(f"    * Large  (≥ 96²)      : {len(grouped_by_size['large'])}")
    print(f"  - Tổng thời gian        : {total_elapsed:.2f} giây")
    print(f"  - Thư mục Ảnh Small     : {os.path.relpath(VIS_SMALL_DIR, BASE_DIR)}/")
    print(f"  - Thư mục Ảnh Medium    : {os.path.relpath(VIS_MEDIUM_DIR, BASE_DIR)}/")
    print(f"  - Thư mục Ảnh Large     : {os.path.relpath(VIS_LARGE_DIR, BASE_DIR)}/")
    print(f"  - File JSON kết quả     : {os.path.relpath(JSON_OUTPUT_PATH, BASE_DIR)}")
    print(f"  - Biểu đồ Boxplot       : {os.path.relpath(BOXPLOT_OUTPUT_PATH, BASE_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
