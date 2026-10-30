import os
import sys
import json
import logging
import importlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

dedup_mod = importlib.import_module('eval_groundtruth.01_deduplicate')
gt_mod = importlib.import_module('eval_groundtruth.02_load_groundtruth')
fmt_mod = importlib.import_module('eval_groundtruth.03_coco_formatter')
eval_mod = importlib.import_module('eval_groundtruth.04_evaluate_map')

apply_wbf_deduplication = dedup_mod.apply_wbf_deduplication
load_groundtruth_yolo = gt_mod.load_groundtruth_yolo
get_image_size = gt_mod.get_image_size
format_coco_dataset = fmt_mod.format_coco_dataset
evaluate_map = eval_mod.evaluate_map

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("eval_groundtruth.run_evaluation")

PRED_HIERARCHY_DIR = os.path.join(BASE_DIR, "combine", "full_pipeline", "results", "json")
RESULTS_DIR = os.path.join(BASE_DIR, "eval_groundtruth", "results")


def run_full_evaluation():
    logger.info("=== BẮT ĐẦU CHẠY ĐÁNH GIÁ mAP50-95 SO VỚI GROUND TRUTH ===")

    pred_files = sorted([f for f in os.listdir(PRED_HIERARCHY_DIR) if f.endswith("_full_hierarchy.json")])
    image_ids = [f.replace("_full_hierarchy.json", "") for f in pred_files]

    images_gt = []
    images_dt = []

    total_gt_boxes = 0
    total_raw_preds = 0
    total_fused_preds = 0

    for img_id in image_ids:
        img_file = f"{img_id}.png"
        img_w, img_h = get_image_size(img_id)

        # 1. Load GT
        gt_boxes = load_groundtruth_yolo(img_id)
        images_gt.append((img_file, img_w, img_h, gt_boxes))
        total_gt_boxes += len(gt_boxes)

        # 2. Load Predictions & Deduplicate via WBF
        pred_path = os.path.join(PRED_HIERARCHY_DIR, f"{img_id}_full_hierarchy.json")
        with open(pred_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)

        raw_elements = pred_data.get("elements", [])
        total_raw_preds += len(raw_elements)

        # Deduplicate
        fused_elements = apply_wbf_deduplication(raw_elements, iou_thresh=0.85)
        images_dt.append((img_file, fused_elements))
        total_fused_preds += len(fused_elements)

    # 3. Format COCO
    coco_gt, coco_dt = format_coco_dataset(images_gt, images_dt)

    # 4. Evaluate mAP
    metrics = evaluate_map(coco_gt, coco_dt)

    # 5. Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.json")

    summary_data = {
        "dataset_size": len(image_ids),
        "total_gt_boxes": total_gt_boxes,
        "total_raw_predictions": total_raw_preds,
        "total_fused_predictions": total_fused_preds,
        "deduplication_iou_threshold": 0.85,
        "metrics": metrics
    }

    with open(out_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("=== KẾT QUẢ ĐÁNH GIÁ CHỈ SỐ mAP50-95 SO VỚI GROUND TRUTH ===")
    print("=" * 70)
    print(f"  - Số ảnh đánh giá             : {len(image_ids)}")
    print(f"  - Tổng số Ground Truth boxes : {total_gt_boxes}")
    print(f"  - Tổng số Raw Predictions     : {total_raw_preds}")
    print(f"  - Số Fused Predictions (WBF)  : {total_fused_preds}")
    print("-" * 70)
    print("  BẢNG CHỈ SỐ mAP COCO EVALUATION:")
    print(f"  * mAP@50-95 (Chỉ số chính)     : {metrics['mAP50_95'] * 100:>6.2f}%")
    print(f"  * mAP@50 (IoU >= 0.50)         : {metrics['mAP50'] * 100:>6.2f}%")
    print(f"  * mAP@75 (IoU >= 0.75)         : {metrics['mAP75'] * 100:>6.2f}%")
    print("=" * 70)
    print(f"File báo cáo JSON chi tiết lưu tại: {out_summary_path}\n")


if __name__ == "__main__":
    run_full_evaluation()
