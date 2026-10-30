import os
import sys
import json
import logging
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from combine._core.schema import load_large
from combine.large_x_large.classify import classify_large_x_large

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("large_x_large.run_pipeline")

LARGE_DIR = os.path.join(BASE_DIR, "yolo", "results", "large")
RESULTS_DIR = os.path.join(BASE_DIR, "combine", "large_x_large", "results", "json")


def process_image(img_id: str) -> Dict[str, Any]:
    clean_id = img_id.replace('.png', '').replace('_large', '')
    large_objs = load_large(clean_id, BASE_DIR)

    relations = classify_large_x_large(large_objs)

    output_data = {
        "image": f"{clean_id}.png",
        "large_count": len(large_objs),
        "relations_count": len(relations),
        "relations": relations
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{clean_id}_large_large.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return output_data


def run_batch():
    logger.info("=== BẮT ĐẦU CHẠY BATCH LARGE x LARGE (30 ÁNH VALIDATION) ===")
    large_files = sorted([f for f in os.listdir(LARGE_DIR) if f.endswith("_large.json")])
    image_ids = [f.replace("_large.json", "") for f in large_files]

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
    print("=== THỐNG KÊ TỔNG KẾT BATCH PROCESSING (LARGE x LARGE) ===")
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
