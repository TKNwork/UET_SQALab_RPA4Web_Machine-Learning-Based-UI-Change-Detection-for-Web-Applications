import os
import sys
import time
import json
import base64
import logging
import requests
import numpy as np
from PIL import Image
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# System logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger("run_full_dataset")

VAL_IMAGES_DIR = os.path.join(BASE_DIR, "data", "yolo_pixelweb_dataset", "images", "val")
VAL_LABELS_DIR = os.path.join(BASE_DIR, "data", "yolo_pixelweb_dataset", "labels", "val")

# Step directories
OCR_RESULTS_DIR = os.path.join(BASE_DIR, "ocr", "results")
YOLO_RESULTS_DIR = os.path.join(BASE_DIR, "yolo", "results")
YOLO_SMALL_DIR = os.path.join(YOLO_RESULTS_DIR, "small")
YOLO_MEDIUM_DIR = os.path.join(YOLO_RESULTS_DIR, "medium")
YOLO_LARGE_DIR = os.path.join(YOLO_RESULTS_DIR, "large")
CV_RESULTS_DIR = os.path.join(BASE_DIR, "cv", "results", "json")
COMBINE_HIERARCHY_DIR = os.path.join(BASE_DIR, "combine", "full_pipeline", "results", "json")
EVAL_RESULTS_DIR = os.path.join(BASE_DIR, "eval_groundtruth", "results")

# Ensure all directories exist
for d in [OCR_RESULTS_DIR, YOLO_RESULTS_DIR, YOLO_SMALL_DIR, YOLO_MEDIUM_DIR, YOLO_LARGE_DIR,
          CV_RESULTS_DIR, COMBINE_HIERARCHY_DIR, EVAL_RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ==============================================================================
# 1. OCR BATCH RUNNER (Google Cloud Vision API)
# ==============================================================================
def run_ocr_batch(image_files: list):
    logger.info("=== STEP 1: CHẠY OCR INFERENCE BATCH (GCV API) ===")
    from ocr.test_gcv_ocr import load_api_key, KEY_FILE, API_ENDPOINT, extract_paragraphs_from_response

    api_key = load_api_key(KEY_FILE)
    success = 0
    skipped = 0

    for idx, img_name in enumerate(image_files, 1):
        clean_id = img_name.replace('.png', '')
        out_json = os.path.join(OCR_RESULTS_DIR, f"{clean_id}.json")

        if os.path.exists(out_json):
            skipped += 1
            continue

        img_path = os.path.join(VAL_IMAGES_DIR, img_name)
        try:
            with open(img_path, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')

            payload = {
                'requests': [{
                    'image': {'content': img_b64},
                    'features': [{'type': 'DOCUMENT_TEXT_DETECTION'}]
                }]
            }

            resp = requests.post(f"{API_ENDPOINT}?key={api_key}", json=payload, timeout=30)
            if resp.status_code == 200:
                res_obj = resp.json().get('responses', [{}])[0]
                paragraphs = extract_paragraphs_from_response(res_obj, conf_threshold=0.80)
                
                json_data = {
                    "image": img_name,
                    "confidence_threshold": 0.80,
                    "paragraphs_count": len(paragraphs),
                    "paragraphs": paragraphs
                }
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)

                success += 1
            else:
                logger.error(f"OCR Error for {img_name}: {resp.status_code} - {resp.text}")

        except Exception as e:
            logger.error(f"Lỗi OCR ảnh {img_name}: {e}")

        if idx % 50 == 0 or idx == len(image_files):
            logger.info(f"  - Tiến độ OCR: {idx}/{len(image_files)} (Đã có sẵn: {skipped}, Mới làm: {success})")

    logger.info(f"-> Hoàn tất OCR batch: {skipped} đã tồn tại, {success} mới xử lý.")


# ==============================================================================
# 2. YOLO BATCH RUNNER (PyTorch / Ultralytics)
# ==============================================================================
def run_yolo_batch(image_files: list):
    logger.info("=== STEP 2: CHẠY YOLO INFERENCE BATCH (3 SIZE) ===")
    from yolo.test_yolo_inference import detect_device, SIZE_THRESHOLDS, predict_and_filter_yolo
    from ultralytics import YOLO

    device = detect_device()
    model_path = os.environ.get('RPA_YOLO_MODEL') or os.path.join(BASE_DIR, 'yolo', 'models', 'yl26best.pt')
    model = YOLO(model_path)

    success = 0
    skipped = 0

    for idx, img_name in enumerate(image_files, 1):
        clean_id = img_name.replace('.png', '')
        main_json = os.path.join(YOLO_RESULTS_DIR, f"{clean_id}.json")

        if os.path.exists(main_json):
            skipped += 1
            continue

        img_path = os.path.join(VAL_IMAGES_DIR, img_name)
        try:
            detections, _ = predict_and_filter_yolo(model, img_path, device=device, size_thresholds=SIZE_THRESHOLDS)
            
            # Ghi main json
            main_content = {
                "image": img_name,
                "size_thresholds": SIZE_THRESHOLDS,
                "detections": detections
            }
            with open(main_json, "w", encoding="utf-8") as f:
                json.dump(main_content, f, ensure_ascii=False, indent=2)

            # Split into 3 size json files
            for cat, folder, conf_thresh in [
                ("small", YOLO_SMALL_DIR, SIZE_THRESHOLDS["small"]),
                ("medium", YOLO_MEDIUM_DIR, SIZE_THRESHOLDS["medium"]),
                ("large", YOLO_LARGE_DIR, SIZE_THRESHOLDS["large"])
            ]:
                cat_dets = [d for d in detections if d.get("size_category") == cat]
                cat_content = {
                    "image": img_name,
                    "size_category": cat,
                    "confidence_threshold": conf_thresh,
                    "detections_count": len(cat_dets),
                    "detections": cat_dets
                }
                cat_json = os.path.join(folder, f"{clean_id}_{cat}.json")
                with open(cat_json, "w", encoding="utf-8") as f:
                    json.dump(cat_content, f, ensure_ascii=False, indent=2)

            success += 1

        except Exception as e:
            logger.error(f"Lỗi YOLO ảnh {img_name}: {e}")

        if idx % 100 == 0 or idx == len(image_files):
            logger.info(f"  - Tiến độ YOLO: {idx}/{len(image_files)} (Đã có sẵn: {skipped}, Mới làm: {success})")

    logger.info(f"-> Hoàn tất YOLO batch: {skipped} đã tồn tại, {success} mới xử lý.")


# ==============================================================================
# 3. CV BATCH RUNNER (OpenCV Contour UI Detector)
# ==============================================================================
def run_cv_batch(image_files: list):
    logger.info("=== STEP 3: CHẠY OPENCV CONTOUR UI DETECTOR BATCH ===")
    from cv.contour_ui_detector import detect_ui_elements

    success = 0
    skipped = 0

    for idx, img_name in enumerate(image_files, 1):
        clean_id = img_name.replace('.png', '')
        cv_json = os.path.join(CV_RESULTS_DIR, f"{clean_id}_cv.json")

        if os.path.exists(cv_json):
            skipped += 1
            continue

        img_path = os.path.join(VAL_IMAGES_DIR, img_name)
        try:
            res = detect_ui_elements(img_path, min_rect_conf=0.90, min_divider_conf=0.85)
            dets = res['detections']
            timings = res['timing_info']

            counts_this_img = {}
            for d in dets:
                lbl = d['label']
                counts_this_img[lbl] = counts_this_img.get(lbl, 0) + 1

            json_data = {
                "image": img_name,
                "image_shape": res['image_shape'],
                "detections_count": len(dets),
                "label_breakdown": counts_this_img,
                "timing_ms": timings,
                "detections": dets
            }
            with open(cv_json, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            success += 1

        except Exception as e:
            logger.error(f"Lỗi CV ảnh {img_name}: {e}")

        if idx % 200 == 0 or idx == len(image_files):
            logger.info(f"  - Tiến độ CV: {idx}/{len(image_files)} (Đã có sẵn: {skipped}, Mới làm: {success})")

    logger.info(f"-> Hoàn tất CV batch: {skipped} đã tồn tại, {success} mới xử lý.")


# ==============================================================================
# 4. PAIRWISE COMBINE MODULES (9 MODULES)
# ==============================================================================
def run_pairwise_combine_modules():
    logger.info("=== STEP 4: CHẠY 9 PAIRWISE COMBINE MODULES ===")
    import importlib

    subdirs = [
        'small_x_cv', 'small_x_yolo', 'medium_x_ocr', 'medium_x_cv',
        'medium_x_large', 'cv_x_large', 'ocr_x_large', 'large_x_large', 'small_x_ocr'
    ]

    for sd in subdirs:
        logger.info(f"  * Running module: combine.{sd}.run_pipeline")
        mod = importlib.import_module(f"combine.{sd}.run_pipeline")
        mod.run_batch()

    logger.info("-> Hoàn tất 9 Pairwise Combine Modules.")


# ==============================================================================
# 5. FULL PIPELINE HIERARCHY BUILDER
# ==============================================================================
def run_full_pipeline_builder(image_files: list):
    logger.info("=== STEP 5: DỰNG CÂY PHÂN CẤP FULL PIPELINE HIERARCHY FOR ALL IMAGES ===")
    from combine.full_pipeline.run_pipeline import process_image

    total_images = len(image_files)
    for idx, img_name in enumerate(image_files, 1):
        process_image(img_name)
        if idx % 100 == 0 or idx == total_images:
            logger.info(f"  - Tiến độ Hierarchy Builder: {idx}/{total_images}")

    logger.info("-> Hoàn tất Full Pipeline Hierarchy Builder cho tất cả các ảnh.")


# ==============================================================================
# 6. COCO mAP EVALUATION (997 IMAGES)
# ==============================================================================
def run_evaluation_all():
    logger.info("=== STEP 6: ĐÁNH GIÁ COCO mAP TRÊN TOÀN BỘ 997 ÁNH GROUND TRUTH ===")
    import importlib

    dedup_mod = importlib.import_module('eval_groundtruth.01_deduplicate')
    gt_mod = importlib.import_module('eval_groundtruth.02_load_groundtruth')
    fmt_mod = importlib.import_module('eval_groundtruth.03_coco_formatter')
    eval_mod = importlib.import_module('eval_groundtruth.04_evaluate_map')

    apply_wbf_deduplication = dedup_mod.apply_wbf_deduplication
    load_groundtruth_yolo = gt_mod.load_groundtruth_yolo
    get_image_size = gt_mod.get_image_size
    format_coco_dataset = fmt_mod.format_coco_dataset
    evaluate_map = eval_mod.evaluate_map

    pred_files = sorted([f for f in os.listdir(COMBINE_HIERARCHY_DIR) if f.endswith("_full_hierarchy.json")])
    image_ids = [f.replace("_full_hierarchy.json", "") for f in pred_files]

    images_gt = []
    images_dt_all = []
    images_dt_yolo = []

    total_gt_boxes = 0
    total_raw_preds = 0
    total_fused_preds = 0
    total_yolo_preds = 0

    for img_id in image_ids:
        img_file = f"{img_id}.png"
        img_w, img_h = get_image_size(img_id)

        # 1. Ground Truth
        gt_boxes = load_groundtruth_yolo(img_id)
        images_gt.append((img_file, img_w, img_h, gt_boxes))
        total_gt_boxes += len(gt_boxes)

        # 2. Predictions & WBF
        pred_path = os.path.join(COMBINE_HIERARCHY_DIR, f"{img_id}_full_hierarchy.json")
        with open(pred_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)

        raw_elements = pred_data.get("elements", [])
        total_raw_preds += len(raw_elements)

        fused_elements = apply_wbf_deduplication(raw_elements, iou_thresh=0.85)
        images_dt_all.append((img_file, fused_elements))
        total_fused_preds += len(fused_elements)

        yolo_elements = [e for e in fused_elements if e.get("source") in ("small", "medium", "large")]
        images_dt_yolo.append((img_file, yolo_elements))
        total_yolo_preds += len(yolo_elements)

    # Format COCO
    coco_gt_all, coco_dt_all = format_coco_dataset(images_gt, images_dt_all)
    metrics_all = evaluate_map(coco_gt_all, coco_dt_all)

    coco_gt_yolo, coco_dt_yolo = format_coco_dataset(images_gt, images_dt_yolo)
    metrics_yolo = evaluate_map(coco_gt_yolo, coco_dt_yolo)

    out_summary_path = os.path.join(EVAL_RESULTS_DIR, "evaluation_summary_997_images.json")
    summary_data = {
        "dataset_size": len(image_ids),
        "total_gt_boxes": total_gt_boxes,
        "total_raw_predictions": total_raw_preds,
        "total_fused_predictions_all": total_fused_preds,
        "total_fused_predictions_yolo": total_yolo_preds,
        "metrics_all_sources": metrics_all,
        "metrics_yolo_only": metrics_yolo
    }

    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print(f"=== BÁO CÁO ĐÁNH GIÁ mAP COCO TRÊN TOÀN BỘ {len(image_ids)} ÁNH GROUND TRUTH ===")
    print("=" * 75)
    print(f"  - Số ảnh đánh giá             : {len(image_ids)}")
    print(f"  - Tổng số Ground Truth boxes : {total_gt_boxes}")
    print(f"  - Tổng số Raw Predictions     : {total_raw_preds}")
    print(f"  - Fused Predictions (All src) : {total_fused_preds}")
    print(f"  - Fused Predictions (YOLO)    : {total_yolo_preds}")
    print("-" * 75)
    print("  1. CHỈ SỐ TOÀN BỘ LUỒNG KẾT HỢP (YOLO + OCR + CV):")
    print(f"     * mAP@50-95 (Chính)        : {metrics_all['mAP50_95'] * 100:>6.2f}%")
    print(f"     * mAP@50                   : {metrics_all['mAP50'] * 100:>6.2f}%")
    print(f"     * mAP@75                   : {metrics_all['mAP75'] * 100:>6.2f}%")
    print("-" * 75)
    print("  2. CHỈ SỐ RIÊNG LUỒNG YOLO (SAU WBF TỈNH CHỈNH):")
    print(f"     * mAP@50-95 (Chính)        : {metrics_yolo['mAP50_95'] * 100:>6.2f}%")
    print(f"     * mAP@50                   : {metrics_yolo['mAP50'] * 100:>6.2f}%")
    print(f"     * mAP@75                   : {metrics_yolo['mAP75'] * 100:>6.2f}%")
    print("=" * 75)
    print(f"Báo cáo chi tiết lưu tại: {out_summary_path}\n")


def main():
    logger.info("=== KÍCH HOẠT QUY TRÌNH TIẾN HÀNH TEST TOÀN BỘ DATASET (997 ÁNH) ===")

    image_files = sorted([
        f for f in os.listdir(VAL_IMAGES_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    logger.info(f"Tìm thấy tổng cộng {len(image_files)} ảnh trong thư mục: {VAL_IMAGES_DIR}")

    t_start = time.time()

    # Step 1: OCR
    run_ocr_batch(image_files)

    # Step 2: YOLO
    run_yolo_batch(image_files)

    # Step 3: CV
    run_cv_batch(image_files)

    # Step 4: 9 Pairwise modules
    run_pairwise_combine_modules()

    # Step 5: Full Pipeline Hierarchy Builder
    run_full_pipeline_builder(image_files)

    # Step 6: Evaluation
    run_evaluation_all()

    total_time = time.time() - t_start
    logger.info(f"=== TOÀN BỘ QUY TRÌNH ĐÃ HOÀN THÀNH XONG TRONG {total_time:.2f} GIÂY (~{total_time/60:.2f} PHÚT) ===")


if __name__ == "__main__":
    main()
