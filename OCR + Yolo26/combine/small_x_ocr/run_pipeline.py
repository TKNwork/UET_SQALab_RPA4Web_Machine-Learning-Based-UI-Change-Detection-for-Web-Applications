import os
import sys
import json
import logging
import argparse
import numpy as np
from typing import List, Dict, Any

# Đảm bảo import được các module nội bộ
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from combine.small_x_ocr.normalize import normalize_yolo_small, normalize_ocr
from combine.small_x_ocr.matrix import compute_matrix
from combine.small_x_ocr.classify import classify_small_boxes

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("run_pipeline")

SMALL_YOLO_DIR = os.path.join(BASE_DIR, "yolo", "results", "small")
OCR_DIR = os.path.join(BASE_DIR, "ocr", "results")
RESULTS_DIR = os.path.join(BASE_DIR, "combine", "small_x_ocr", "results", "json")


def process_single_image(img_id: str) -> Dict[str, Any]:
    """
    Xử lý end-to-end cho 1 ảnh (ví dụ img_id = 'b1_1020')
    """
    # Đường dẫn file
    yolo_small_path = os.path.join(SMALL_YOLO_DIR, f"{img_id}_small.json")
    ocr_path = os.path.join(OCR_DIR, f"{img_id}.json")

    # 1. Normalize
    small_objs = normalize_yolo_small(yolo_small_path)
    ocr_objs = normalize_ocr(ocr_path)

    # Chuyển đổi tọa độ bbox thành numpy arrays
    small_boxes = np.array([obj['bbox'] for obj in small_objs], dtype=np.float64) if small_objs else np.empty((0, 4), dtype=np.float64)
    ocr_boxes = np.array([obj['bbox'] for obj in ocr_objs], dtype=np.float64) if ocr_objs else np.empty((0, 4), dtype=np.float64)

    # 2. Matrix calculation
    matrix_dict = compute_matrix(small_boxes, ocr_boxes)

    # 3. Classify
    matches = classify_small_boxes(small_objs, ocr_objs, matrix_dict)

    # Formatter tên ảnh (vd "b1_1020.png")
    image_file_name = f"{img_id}.png"

    output_data = {
        "image": image_file_name,
        "matches": matches
    }

    # 4. Save JSON result
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_json_path = os.path.join(RESULTS_DIR, f"{img_id}_small_ocr_match.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return output_data


def run_single_test(img_id: str = "b1_1020"):
    logger.info(f"=== TEST THỬ NGHIỆM TRÊN ẤN ĐỊNH: {img_id} ===")
    res = process_single_image(img_id)

    print("\n" + "=" * 60)
    print(f"KẾT QUẢ SO KHỚP SMALL YOLO x OCR - CHUYÊN SÂU TRÊN AMH {img_id}:")
    print("=" * 60)
    for m in res["matches"]:
        small_id = m["small_id"]
        case = m["case"]
        ocr_match = m["primary_ocr_match"]
        metrics = m["metrics"]
        print(f"- {small_id} | Box: {m['bbox']} | Conf: {m['confidence']}")
        print(f"  -> Case: {case}")
        print(f"  -> Match Text: {ocr_match}")
        print(f"  -> Metrics: {metrics}")
        print("-" * 60)

    # Validation assertions cho b1_1020
    small_3_match = next((m for m in res["matches"] if m["small_id"] == "small_3"), None)
    if small_3_match:
        assert small_3_match["case"] == "duplicate_ocr", f"Expected small_3 to be duplicate_ocr, got {small_3_match['case']}"
        assert small_3_match["primary_ocr_match"] == "Next", f"Expected Next, got {small_3_match['primary_ocr_match']}"
        logger.info("VALIDATION SUCCESS: small_3 ('Next') khớp chính xác 'duplicate_ocr' với IoU ~ 0.978!")

    logger.info(f"Đã lưu file kết quả test: {os.path.join(RESULTS_DIR, f'{img_id}_small_ocr_match.json')}\n")


def run_batch():
    logger.info("=== BẮT ĐẦU CHẠY BATCH TOÀN BỘ DATASET VALIDATION ===")

    # Lấy danh sách file ảnh dựa trên small json files
    small_files = sorted([f for f in os.listdir(SMALL_YOLO_DIR) if f.endswith("_small.json")])
    image_ids = [f.replace("_small.json", "") for f in small_files]

    total_images = len(image_ids)
    total_small_boxes = 0
    case_counts = {
        "duplicate_ocr": 0,
        "nested_in_ocr": 0,
        "partial_overlap_review": 0,
        "standalone_pending_yolo_check": 0
    }

    for idx, img_id in enumerate(image_ids, 1):
        res = process_single_image(img_id)
        matches = res["matches"]
        total_small_boxes += len(matches)

        for m in matches:
            c = m["case"]
            case_counts[c] = case_counts.get(c, 0) + 1

    print("\n" + "=" * 70)
    print("=== THỐNG KÊ TỔNG KẾT BATCH PROCESSING (SMALL YOLO x OCR) ===")
    print("=" * 70)
    print(f"  - Tổng số ảnh đã xử lý    : {total_images}")
    print(f"  - Tổng số Small YOLO boxes: {total_small_boxes}")
    print("-" * 70)
    print("  PHÂN PHỐI CÁC CASE (SỐ LƯỢNG & TỶ LỆ %):")

    for case_name, count in case_counts.items():
        pct = (count / total_small_boxes * 100) if total_small_boxes > 0 else 0.0
        print(f"  * {case_name:<30}: {count:>4} boxes ({pct:>6.2f}%)")

    print("=" * 70)
    print(f"File kết quả JSON từng ảnh lưu tại: {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy pipeline so khớp Small YOLO x OCR")
    parser.add_argument("--test", action="store_true", help="Chạy test kiểm tra duy nhất trên b1_1020")
    args = parser.parse_args()

    if args.test:
        run_single_test("b1_1020")
    else:
        # 1. Luôn chạy test b1_1020 trước để đảm bảo chính xác
        run_single_test("b1_1020")
        # 2. Chạy batch toàn bộ dataset
        run_batch()
