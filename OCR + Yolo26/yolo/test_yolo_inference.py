import os
import sys
import time
import json
from PIL import Image, ImageDraw, ImageFont

# Xác định thư mục gốc của project (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.environ.get('RPA_YOLO_MODEL') or os.path.join(BASE_DIR, 'yolo', 'models', 'yl26best.pt')

VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'yolo_pixelweb_dataset', 'images', 'val')
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'yolo_pixelweb_dataset', 'images', 'val')

YOLO_RESULTS_DIR = os.path.join(BASE_DIR, 'yolo', 'results')
VISUALIZED_DIR = os.path.join(YOLO_RESULTS_DIR, 'visualized')

# Cấu hình ngưỡng confidence theo từng size_category mới:
# large >= 0.85, med >= 0.50, small >= 0.10
SIZE_THRESHOLDS = {
    'small': 0.10,
    'medium': 0.50,
    'large': 0.85
}


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


def get_first_30_images(images_dir, n=30):
    if not os.path.exists(images_dir):
        print(f"[LỖI] Thư mục ảnh '{images_dir}' không tồn tại.")
        sys.exit(1)
        
    files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'))
    ])
    
    return files[:n]


def get_size_category(area):
    if area < 1024:
        return 'small'
    elif area < 9216:
        return 'medium'
    else:
        return 'large'


def predict_and_filter_yolo(model, image_path, device='cpu', size_thresholds=None):
    """
    Chạy suy luận YOLO và áp dụng bộ lọc ngưỡng confidence theo kích thước COCO (large>=0.85, med>=0.50, small>=0.10)
    """
    if size_thresholds is None:
        size_thresholds = SIZE_THRESHOLDS

    start_time = time.time()
    results = model.predict(source=image_path, device=device, conf=0.10, verbose=False)
    elapsed_time = time.time() - start_time

    result = results[0]
    boxes = result.boxes

    detections = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            bbox = [round(coord, 2) for coord in xyxy]
            x_min, y_min, x_max, y_max = bbox
            area = round(max(0, x_max - x_min) * max(0, y_max - y_min), 2)
            size_cat = get_size_category(area)
            confidence = round(float(box.conf[0]), 4)
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, str(cls_id)) if hasattr(model, 'names') else str(cls_id)

            thresh = size_thresholds.get(size_cat, 0.10)
            if confidence >= thresh:
                detections.append({
                    "class": cls_name,
                    "confidence": confidence,
                    "bbox": bbox,
                    "area": area,
                    "size_category": size_cat
                })

    return detections, elapsed_time


def draw_yolo_detections(image_path, detections, output_image_path):
    """Vẽ bbox và label (class name + confidence) lên ảnh gốc"""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    for det in detections:
        bbox = det.get('bbox', [])
        cls_name = det.get('class', 'object')
        conf = det.get('confidence', 0.0)
        
        if len(bbox) == 4:
            x_min, y_min, x_max, y_max = bbox
            draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=2)
            
            label = f"{cls_name} {conf:.2f}"
            text_pos = (x_min, max(0, y_min - 12))
            
            text_bbox = draw.textbbox(text_pos, label, font=font) if hasattr(draw, 'textbbox') else None
            if text_bbox:
                draw.rectangle(text_bbox, fill=(0, 0, 0))
            draw.text(text_pos, label, fill=(255, 255, 0), font=font)

    img.save(output_image_path)


_YOLO_MODEL_CACHE = None


def get_yolo_model(model_path=None, device=None):
    global _YOLO_MODEL_CACHE
    if _YOLO_MODEL_CACHE is not None:
        return _YOLO_MODEL_CACHE
    if model_path is None:
        model_path = MODEL_PATH
        if not os.path.exists(model_path):
            fallback_model = os.path.join(BASE_DIR, 'yl26best.pt')
            if os.path.exists(fallback_model):
                model_path = fallback_model
            else:
                print(f"ERROR: YOLO model not found: {MODEL_PATH}")
                return None
    try:
        from ultralytics import YOLO
        _YOLO_MODEL_CACHE = YOLO(model_path)
        return _YOLO_MODEL_CACHE
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model: {e}")
        return None


def run_yolo(image_path, model_path=None, device=None):
    """
    Hàm reusable chạy YOLO suy luận cho 1 ảnh và trả về 3 danh sách (small, medium, large).
    """
    if device is None:
        device = detect_device()
    model = get_yolo_model(model_path, device)
    if model is None:
        print(f"ERROR: YOLO model not available: {model_path or MODEL_PATH}")
        return [], [], []

    detections, _ = predict_and_filter_yolo(model, image_path, device=device, size_thresholds=SIZE_THRESHOLDS)
    small_dets = [d for d in detections if d.get('size_category') == 'small']
    medium_dets = [d for d in detections if d.get('size_category') == 'medium']
    large_dets = [d for d in detections if d.get('size_category') == 'large']
    return small_dets, medium_dets, large_dets


def main():
    print(f"=== BẮT ĐẦU TEST YOLO INFERENCE (LỌC THEO SIZE THRESHOLDS: Large>=0.85, Med>=0.50, Small>=0.10) ===")
    
    device = detect_device()
    
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        fallback_model = os.path.join(BASE_DIR, 'yl26best.pt')
        if os.path.exists(fallback_model):
            model_path = fallback_model
        else:
            print(f"[LỖI] File weights YOLO '{model_path}' không tồn tại.")
            sys.exit(1)
        
    print(f"Đang nạp mô hình YOLO từ: {model_path}")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
    except Exception as e:
        print(f"[LỖI CRITICAL] Không thể nạp mô hình YOLO: {e}")
        sys.exit(1)
        
    images = get_first_30_images(VAL_IMAGES_DIR, n=30)
    print(f"Đã tìm thấy {len(images)} ảnh đầu tiên trong '{VAL_IMAGES_DIR}':")
    for img in images:
        print(f" - {img}")
    print("-" * 60)
    
    small_dir = os.path.join(YOLO_RESULTS_DIR, 'small')
    medium_dir = os.path.join(YOLO_RESULTS_DIR, 'medium')
    large_dir = os.path.join(YOLO_RESULTS_DIR, 'large')

    os.makedirs(YOLO_RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZED_DIR, exist_ok=True)
    os.makedirs(small_dir, exist_ok=True)
    os.makedirs(medium_dir, exist_ok=True)
    os.makedirs(large_dir, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    total_start_time = time.time()
    
    for idx, img_name in enumerate(images, 1):
        img_path = os.path.join(VAL_IMAGES_DIR, img_name)
        base_name, _ = os.path.splitext(img_name)
        
        json_output_path = os.path.join(YOLO_RESULTS_DIR, f"{base_name}.json")
        pred_img_name = f"{base_name}_pred.png"
        visualized_output_path = os.path.join(VISUALIZED_DIR, pred_img_name)
        
        print(f"[{idx}/{len(images)}] Tên file ảnh: {img_name}")
        
        try:
            detections, elapsed_time = predict_and_filter_yolo(model, img_path, device=device, size_thresholds=SIZE_THRESHOLDS)
            success_count += 1
            
            # File JSON tổng hợp
            json_content = {
                "image": img_name,
                "size_thresholds": SIZE_THRESHOLDS,
                "detections": detections
            }
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(json_content, f, ensure_ascii=False, indent=2)
                
            # Phân tách và ghi 3 file JSON riêng theo size COCO
            for cat, folder, conf_thresh in [
                ("small", small_dir, SIZE_THRESHOLDS["small"]),
                ("medium", medium_dir, SIZE_THRESHOLDS["medium"]),
                ("large", large_dir, SIZE_THRESHOLDS["large"])
            ]:
                cat_dets = [d for d in detections if d.get("size_category") == cat]
                cat_content = {
                    "image": img_name,
                    "size_category": cat,
                    "confidence_threshold": conf_thresh,
                    "detections_count": len(cat_dets),
                    "detections": cat_dets
                }
                cat_file_path = os.path.join(folder, f"{base_name}_{cat}.json")
                with open(cat_file_path, "w", encoding="utf-8") as f:
                    json.dump(cat_content, f, ensure_ascii=False, indent=2)

            draw_yolo_detections(img_path, detections, visualized_output_path)
            
            print(f"  - Số object detect được (lọc theo size): {len(detections)}")
            print(f"  - File JSON lưu tại   : {os.path.relpath(json_output_path, BASE_DIR)}")
            print(f"  - Thời gian xử lý     : {elapsed_time:.3f} giây")
            print("-" * 60)
            
        except Exception as e:
            fail_count += 1
            print(f"  - [LỖI]: Không thể xử lý ảnh '{img_name}': {e}")
            print("-" * 60)
            
    total_elapsed_time = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("=== TỔNG KẾT QUÁ TRÌNH YOLO INFERENCE ===")
    print(f"  - Size Thresholds      : {SIZE_THRESHOLDS}")
    print(f"  - Số ảnh thành công     : {success_count}/{len(images)}")
    print(f"  - Số ảnh thất bại       : {fail_count}/{len(images)}")
    print(f"  - Tổng thời gian        : {total_elapsed_time:.2f} giây")
    print(f"  - Thư mục JSON          : {os.path.relpath(YOLO_RESULTS_DIR, BASE_DIR)}/")
    print(f"  - Thư mục Visualized    : {os.path.relpath(VISUALIZED_DIR, BASE_DIR)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
