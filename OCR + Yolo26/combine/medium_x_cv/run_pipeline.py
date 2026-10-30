import os
import sys
import json
import logging
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from combine._core.schema import load_medium, load_cv
from combine.medium_x_cv.classify import classify_medium_x_cv

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("medium_x_cv.run_pipeline")

MEDIUM_DIR = os.path.join(BASE_DIR, "yolo", "results", "medium")
RESULTS_DIR = os.path.join(BASE_DIR, "combine", "medium_x_cv", "results", "json")


def process_image(img_id: str) -> Dict[str, Any]:
    clean_id = img_id.replace('.png', '').replace('_medium', '')
    medium_objs = load_medium(clean_id, BASE_DIR)
    cv_objs = load_cv(clean_id, BASE_DIR)

    relations = classify_medium_x_cv(medium_objs, cv_objs)

    output_data = {
        "image": f"{clean_id}.png",
        "medium_count": len(medium_objs),
        "cv_count": len(cv_objs),
        "relations_count": len(relations),
        "relations": relations
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{clean_id}_medium_cv.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return output_data


def run_batch():
    logger.info("=== BẮT ĐẦU CHẠY BATCH MEDIUM x CV (30 ÁNH VALIDATION) ===")
    medium_files = sorted([f for f in os.listdir(MEDIUM_DIR) if f.endswith("_medium.json")])
    image_ids = [f.replace("_medium.json", "") for f in medium_files]

    total_images = len(image_ids)
    total_relations = 0
    case_counts = {}

    for img_id in image_ids:
        res = process_image(img_id)
        rels = res["relations"]
        total_relations += len(rels)
        for r in rels:
            c = r["case"]
            case_counts[c] = case_counts.get(c, 0) + 1

    print("\n" + "=" * 70)
    print("=== THỐNG KÊ TỔNG KẾT BATCH PROCESSING (MEDIUM x CV) ===")
    print("=" * 70)
    print(f"  - Tổng số ảnh đã xử lý  : {total_images}")
    print(f"  - Tổng số quan hệ phát hiện: {total_relations}")
    print("-" * 70)
    print("  PHÂN PHỐI CÁC CASE QUAN HỆ (SỐ LƯỢNG & TỶ LỆ %):")

    for case_name, count in case_counts.items():
        pct = (count / total_relations * 100) if total_relations > 0 else 0.0
        print(f"  * {case_name:<25}: {count:>4} pairs ({pct:>6.2f}%)")

    print("=" * 70)
    print(f"File kết quả JSON lưu tại: {RESULTS_DIR}\n")


if __name__ == "__main__":
    run_batch()
