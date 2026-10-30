import os
import sys
import time
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple, Union

# Xác định thư mục gốc của project (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Thư mục dữ liệu ảnh test
VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'yolo_pixelweb_dataset', 'images', 'val')
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'yolo_pixelweb_dataset', 'images', 'val')

# Thư mục lưu kết quả CV
CV_DIR = os.path.join(BASE_DIR, 'cv')
TEST_IMAGES_DIR = os.path.join(CV_DIR, 'test_images')
OUTPUT_DIR = os.path.join(CV_DIR, 'output')
RESULTS_JSON_DIR = os.path.join(CV_DIR, 'results', 'json')
RESULTS_VIS_DIR = os.path.join(CV_DIR, 'results', 'visualized')


def apply_nms(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Áp dụng Non-Max Suppression (NMS) dựa trên chỉ số IoU
    để loại bỏ các bounding box bị chồng lấp quá mức.
    
    :param detections: Danh sách các dict detection (bbox: [x,y,w,h], confidence, label...)
    :param iou_threshold: Ngưỡng IoU tối đa cho phép chồng lấp (default = 0.5)
    :return: Danh sách detections đã qua lọc NMS
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.get('confidence', 0.0), reverse=True)
    keep_dets = []
    
    while sorted_dets:
        best_det = sorted_dets.pop(0)
        keep_dets.append(best_det)

        bx1, by1, bw1, bh1 = best_det['bbox']
        x1_max, y1_max = bx1 + bw1, by1 + bh1
        area1 = bw1 * bh1

        remaining = []
        for det in sorted_dets:
            bx2, by2, bw2, bh2 = det['bbox']
            x2_max, y2_max = bx2 + bw2, by2 + bh2
            area2 = bw2 * bh2

            inter_x1 = max(bx1, bx2)
            inter_y1 = max(by1, by2)
            inter_x2 = min(x1_max, x2_max)
            inter_y2 = min(y1_max, y2_max)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            union_area = area1 + area2 - inter_area
            iou = inter_area / float(union_area) if union_area > 0 else 0.0

            if iou < iou_threshold:
                remaining.append(det)

        sorted_dets = remaining

    return keep_dets


def detect_ui_elements(
    image_input: Union[str, np.ndarray],
    conf_threshold: float = 0.1,
    iou_threshold: float = 0.5,
    min_rect_conf: float = 0.90,     # Confidence tối thiểu cho rectangle_box >= 0.90
    min_divider_conf: float = 0.85   # Confidence tối thiểu cho divider >= 0.85
) -> Dict:
    """
    Phát hiện các phần tử UI từ ảnh screenshot bằng Computer Vision thuần túy (OpenCV).
    - Lọc 'rectangle_box' với confidence >= 0.90 (Đã bỏ lọc diện tích >= 40^2)
    - Lọc 'divider' với điểm confidence >= 0.85

    :param image_input: Đường dẫn tới file ảnh (str) hoặc mảng numpy ndarray (BGR)
    :param conf_threshold: Ngưỡng confidence tối thiểu chung
    :param iou_threshold: Ngưỡng IoU dùng cho NMS
    :param min_rect_conf: Ngưỡng confidence tối thiểu cho rectangle_box (default = 0.90)
    :param min_divider_conf: Ngưỡng confidence tối thiểu cho divider (default = 0.85)
    :return: Dict chứa 'detections', 'timing_info', 'image_shape'
    """
    t_start_total = time.perf_counter()
    timing_info = {}

    # 1. Đọc và nạp ảnh
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Không thể mở hoặc decode file ảnh: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError("image_input phải là đường dẫn file (str) hoặc np.ndarray")

    h_img, w_img = img.shape[:2]

    # 2. Bước 1: Chuyển ảnh xám (Grayscale)
    t0 = time.perf_counter()
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    t1 = time.perf_counter()
    timing_info['grayscale_time_ms'] = round((t1 - t0) * 1000, 3)

    # 3. Bước 2: Canny Edge Detection với Otsu Auto-thresholding
    t0 = time.perf_counter()
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower_canny = 0.5 * otsu_val
    upper_canny = otsu_val
    edges = cv2.Canny(gray, lower_canny, upper_canny)
    t1 = time.perf_counter()
    timing_info['canny_edge_time_ms'] = round((t1 - t0) * 1000, 3)

    # 4. Bước 3: Dilation nhẹ (kernel 3x3, 1 iteration) để nối các cạnh bị đứt nét
    t0 = time.perf_counter()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    t1 = time.perf_counter()
    timing_info['dilation_time_ms'] = round((t1 - t0) * 1000, 3)

    # 5. Bước 4: Tìm contours sử dụng RETR_TREE + CHAIN_APPROX_SIMPLE
    t0 = time.perf_counter()
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    t1 = time.perf_counter()
    timing_info['find_contours_time_ms'] = round((t1 - t0) * 1000, 3)

    # 6. Bước 5: Trích xuất các nhãn Divider (conf >= 0.85) và Rectangle Box (conf >= 0.90)
    t0 = time.perf_counter()
    raw_detections = []
    
    if hierarchy is not None and len(contours) > 0:
        hierarchy_arr = hierarchy[0]

        for i, cnt in enumerate(contours):
            next_idx, prev_idx, child_idx, parent_idx = hierarchy_arr[i]

            x, y, w, h = cv2.boundingRect(cnt)
            if w >= w_img * 0.99 and h >= h_img * 0.99:
                continue

            contour_area = cv2.contourArea(cnt)
            bbox_area = w * h
            if contour_area < 20 or bbox_area < 20:
                continue

            if parent_idx != -1:
                px, py, pw, ph = cv2.boundingRect(contours[parent_idx])
                if abs(x - px) < 5 and abs(y - py) < 5 and abs(w - pw) < 5 and abs(h - ph) < 5:
                    continue

            aspect_ratio = round(float(w) / float(h), 4) if h > 0 else 0.0
            extent = round(float(contour_area) / float(bbox_area), 4) if bbox_area > 0 else 0.0

            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            n_vertices = len(approx)

            # 1. Phân loại Divider (aspect_ratio >= 8.0 hoặc <= 0.125 VÀ conf >= 0.85)
            if aspect_ratio >= 8.0 or (aspect_ratio <= 0.125 and aspect_ratio > 0):
                confidence = min(1.0, max(0.6, aspect_ratio / 12.0 if aspect_ratio >= 8.0 else (1.0 / aspect_ratio) / 12.0))
                if confidence >= min_divider_conf:
                    raw_detections.append({
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "label": "divider",
                        "confidence": round(confidence, 4),
                        "n_vertices": int(n_vertices),
                        "aspect_ratio": aspect_ratio,
                        "extent": extent,
                        "area": float(contour_area)
                    })

            # 2. Phân loại Rectangle Box (n_vertices == 4, extent >= 0.75 VÀ conf >= 0.90)
            elif n_vertices == 4 and extent >= 0.75:
                confidence = min(1.0, round(extent, 4))
                if confidence >= min_rect_conf:
                    raw_detections.append({
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "label": "rectangle_box",
                        "confidence": round(confidence, 4),
                        "n_vertices": int(n_vertices),
                        "aspect_ratio": aspect_ratio,
                        "extent": extent,
                        "area": float(contour_area)
                    })

    t1 = time.perf_counter()
    timing_info['feature_extraction_time_ms'] = round((t1 - t0) * 1000, 3)

    # 7. Bước 6: Non-Max Suppression (NMS)
    t0 = time.perf_counter()
    nms_detections = apply_nms(raw_detections, iou_threshold=iou_threshold)
    t1 = time.perf_counter()
    timing_info['nms_time_ms'] = round((t1 - t0) * 1000, 3)

    t_end_total = time.perf_counter()
    timing_info['total_time_ms'] = round((t_end_total - t_start_total) * 1000, 3)

    return {
        "detections": nms_detections,
        "timing_info": timing_info,
        "image_shape": [h_img, w_img]
    }


def draw_detections(
    image_input: Union[str, np.ndarray],
    detections: List[Dict],
    output_path: str = None
) -> np.ndarray:
    """
    Vẽ các bounding box (divider và rectangle_box) lên ảnh gốc để kiểm tra trực quan.

    :param image_input: Đường dẫn ảnh (str) hoặc numpy ndarray (BGR)
    :param detections: Danh sách dict detections
    :param output_path: Đường dẫn lưu ảnh kết quả (nếu có)
    :return: Ảnh BGR đã được vẽ bbox
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = image_input.copy()

    label_colors = {
        "divider": (0, 165, 255),        # Cam đậm (Orange)
        "rectangle_box": (255, 100, 0)   # Xanh lam (Blue/Cyan)
    }

    for det in detections:
        x, y, w, h = det['bbox']
        label = det.get('label', 'rectangle_box')
        conf = det.get('confidence', 0.0)

        color = label_colors.get(label, (255, 100, 0))

        # Vẽ bounding box (viền 2px)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Vẽ label + confidence
        caption = f"{label} {conf:.2f}"
        font_scale = 0.45
        thickness = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        txt_y1 = max(0, y - txt_h - 4)
        cv2.rectangle(img, (x, txt_y1), (x + txt_w + 4, txt_y1 + txt_h + 4), (30, 30, 30), -1)
        cv2.putText(img, caption, (x + 2, txt_y1 + txt_h + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, img)

    return img


def create_synthetic_ui_image(save_path: str) -> np.ndarray:
    """
    Tự tạo 1 ảnh UI giả lập synthetic để test nhanh pipeline.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Canvas màu nền trắng kem sáng (800x600)
    canvas = np.ones((600, 800, 3), dtype=np.uint8) * 245

    # Navigation bar
    cv2.rectangle(canvas, (20, 20), (780, 70), (200, 200, 200), -1)
    cv2.rectangle(canvas, (20, 20), (780, 70), (100, 100, 100), 2)
    cv2.putText(canvas, "Header Navigation Bar", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

    # Horizontal Divider
    cv2.line(canvas, (20, 90), (780, 90), (150, 150, 150), 2)

    # Input Box 1
    cv2.rectangle(canvas, (60, 160), (430, 200), (230, 230, 230), -1)
    cv2.rectangle(canvas, (60, 160), (430, 200), (80, 80, 80), 2)
    cv2.putText(canvas, "Username Input...", (75, 187), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    # Submit Button
    cv2.rectangle(canvas, (60, 340), (220, 385), (0, 120, 240), -1)
    cv2.rectangle(canvas, (60, 340), (220, 385), (0, 80, 180), 2)
    cv2.putText(canvas, "SUBMIT", (100, 368), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Vertical Divider
    cv2.line(canvas, (480, 120), (480, 540), (150, 150, 150), 2)

    cv2.imwrite(save_path, canvas)
    return canvas


def run_synthetic_test():
    """Chạy test sanity check trên ảnh synthetic UI"""
    print("\n--- 1. CHẠY TEST SANITY CHECK TRÊN ẢNH SYNTHETIC UI ---")
    synth_path = os.path.join(TEST_IMAGES_DIR, 'synthetic_ui.png')
    create_synthetic_ui_image(synth_path)
    print(f"-> Đã tạo ảnh synthetic test tại: {os.path.relpath(synth_path, BASE_DIR)}")

    res = detect_ui_elements(synth_path, min_rect_conf=0.90, min_divider_conf=0.85)
    dets = res['detections']
    timings = res['timing_info']

    print(f"-> Số phần tử detect được trên ảnh synthetic: {len(dets)}")
    print(f"-> Thời gian xử lý CPU (ms): {timings}")

    assert len(dets) > 0, "LỖI SANITY CHECK: Không detect được phần tử nào trên ảnh synthetic UI!"
    print("✅ Sanity check 1 PASSED: Synthetic UI detect được > 0 phần tử.")

    blank_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    res_blank = detect_ui_elements(blank_img)
    print(f"-> Số phần tử detect được trên ảnh trắng trơn: {len(res_blank['detections'])}")
    assert len(res_blank['detections']) == 0, "LỖI: Ảnh trắng trơn không được trả về bbox nhiễu!"
    print("✅ Sanity check 2 PASSED: Ảnh trắng trơn không crash và trả về 0 bbox.")

    synth_out_path = os.path.join(OUTPUT_DIR, 'detection_result.png')
    draw_detections(synth_path, dets, synth_out_path)
    print(f"-> Đã lưu ảnh kết quả synthetic tại: {os.path.relpath(synth_out_path, BASE_DIR)}")


def run_cv(image_path, min_rect_conf=0.90, min_divider_conf=0.85):
    """
    Hàm reusable chạy CV Contour Detection cho 1 ảnh.
    Trả về danh sách detections (mỗi item có bbox [x,y,w,h], label, confidence).
    """
    try:
        res = detect_ui_elements(image_path, min_rect_conf=min_rect_conf, min_divider_conf=min_divider_conf)
        return res.get('detections', [])
    except Exception as e:
        print(f"WARNING: CV detection failed: {e}")
        return []


def main():
    print("=== CV CONTOUR UI DETECTOR (BỎ LỌC DIỆN TÍCH 40^2 - CHỈ GIỮ RECTANGLE CONF >= 0.90 & DIVIDER CONF >= 0.85) ===")


    # 1. Tạo các thư mục
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_JSON_DIR, exist_ok=True)
    os.makedirs(RESULTS_VIS_DIR, exist_ok=True)

    # 2. Chạy test Synthetic UI
    run_synthetic_test()

    # 3. Chạy test trên 30 ảnh trong dataset
    if not os.path.exists(VAL_IMAGES_DIR):
        print(f"\n[CẢNH BÁO] Không tìm thấy thư mục dataset: {VAL_IMAGES_DIR}")
        return

    images = sorted([
        f for f in os.listdir(VAL_IMAGES_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'))
    ])[:30]

    print(f"\n--- 2. CHẠY CONTOUR DETECTION TRÊN {len(images)} ÅNH DATASET VAL ---")
    print("-" * 75)

    label_totals = {
        "rectangle_box": 0,
        "divider": 0
    }

    step_time_totals = {
        "grayscale": 0.0,
        "canny": 0.0,
        "dilation": 0.0,
        "find_contours": 0.0,
        "feature_extraction": 0.0,
        "nms": 0.0,
        "total": 0.0
    }

    start_all = time.perf_counter()

    for idx, img_name in enumerate(images, 1):
        img_path = os.path.join(VAL_IMAGES_DIR, img_name)
        base_name, _ = os.path.splitext(img_name)

        try:
            res = detect_ui_elements(img_path, min_rect_conf=0.90, min_divider_conf=0.85)
            dets = res['detections']
            timings = res['timing_info']

            counts_this_img = {}
            for d in dets:
                lbl = d['label']
                label_totals[lbl] = label_totals.get(lbl, 0) + 1
                counts_this_img[lbl] = counts_this_img.get(lbl, 0) + 1

            step_time_totals['grayscale'] += timings.get('grayscale_time_ms', 0)
            step_time_totals['canny'] += timings.get('canny_edge_time_ms', 0)
            step_time_totals['dilation'] += timings.get('dilation_time_ms', 0)
            step_time_totals['find_contours'] += timings.get('find_contours_time_ms', 0)
            step_time_totals['feature_extraction'] += timings.get('feature_extraction_time_ms', 0)
            step_time_totals['nms'] += timings.get('nms_time_ms', 0)
            step_time_totals['total'] += timings.get('total_time_ms', 0)

            # Lưu file JSON kết quả
            json_out = os.path.join(RESULTS_JSON_DIR, f"{base_name}_cv.json")
            json_data = {
                "image": img_name,
                "image_shape": res['image_shape'],
                "detections_count": len(dets),
                "label_breakdown": counts_this_img,
                "timing_ms": timings,
                "detections": dets
            }
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            # Lưu ảnh vẽ bounding box
            vis_out = os.path.join(RESULTS_VIS_DIR, f"{base_name}_cv.png")
            draw_detections(img_path, dets, vis_out)

            lbl_str = ", ".join([f"{k}: {v}" for k, v in counts_this_img.items()])
            print(f"[{idx}/{len(images)}] {img_name} -> {len(dets)} boxes ({lbl_str}) | Time: {timings['total_time_ms']:.2f}ms")

        except Exception as e:
            print(f"[CẢNH BÁO] Lỗi khi xử lý {img_name}: {e}")

    total_wall_time = (time.perf_counter() - start_all)

    # 4. In bảng tổng kết
    print("\n" + "=" * 75)
    print("TỔNG KẾT KẾT QUẢ DETECT VÀ THỜI GIAN XỬ LÝ (CPU BENCHMARK)")
    print("=" * 75)
    print("1. THỐNG KÊ SỐ LƯỢNG BBOX DETECT ĐƯỢC THEO LOẠI PHẦN TỬ:")
    for lbl, count in label_totals.items():
        print(f"   - {lbl:<15}: {count} boxes")
    print(f"   * TỔNG CỘNG       : {sum(label_totals.values())} boxes trên {len(images)} ảnh")

    print("\n2. THỜI GIAN XỬ LÝ TRUNG BÌNH MỖI BƯỚC PIPELINE TRÊN CPU (WALL-CLOCK TIME):")
    n_img = max(1, len(images))
    print(f"   - Grayscale Conversion  : {step_time_totals['grayscale'] / n_img:.3f} ms / ảnh")
    print(f"   - Canny Edge (Otsu)     : {step_time_totals['canny'] / n_img:.3f} ms / ảnh")
    print(f"   - Dilation (3x3 Kernel) : {step_time_totals['dilation'] / n_img:.3f} ms / ảnh")
    print(f"   - Find Contours (TREE)  : {step_time_totals['find_contours'] / n_img:.3f} ms / ảnh")
    print(f"   - Feature Extraction    : {step_time_totals['feature_extraction'] / n_img:.3f} ms / ảnh")
    print(f"   - NMS (Non-Max Suppr.)  : {step_time_totals['nms'] / n_img:.3f} ms / ảnh")
    print(f"   ---------------------------------------------------")
    print(f"   * Tổng thời gian TB/ảnh : {step_time_totals['total'] / n_img:.3f} ms (~{1000 / (step_time_totals['total'] / n_img):.1f} FPS trên CPU)")
    print(f"   * Tổng wall-clock time  : {total_wall_time:.2f} giây")

    print("\n3. THƯ MỤC KẾT QUẢ:")
    print(f"   - Ảnh test synthetic : {os.path.relpath(OUTPUT_DIR, BASE_DIR)}/detection_result.png")
    print(f"   - JSON 30 ảnh        : {os.path.relpath(RESULTS_JSON_DIR, BASE_DIR)}/")
    print(f"   - Ảnh Visualized 30  : {os.path.relpath(RESULTS_VIS_DIR, BASE_DIR)}/")
    print("=" * 75)


if __name__ == "__main__":
    main()
