import os
import sys
import json
import logging
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from combine._core.schema import load_small, load_medium, load_large
from combine.small_x_yolo.classify import classify_small_x_yolo

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("small_x_yolo.run_pipeline")

SMALL_DIR = os.path.join(BASE_DIR, "yolo", "results", "small")
RESULTS_DIR = os.path.join(BASE_DIR, "combine", "small_x_yolo", "results", "json")


def process_image(img_id: str) -> Dict[str, Any]:
    clean_id = img_id.replace('.png', '').replace('_small', '')
    small_objs = load_small(clean_id, BASE_DIR)
    medium_objs = load_medium(clean_id, BASE_DIR)
    large_objs = load_large(clean_id, BASE_DIR)

    rels_sm, rels_sl = classify_small_x_yolo(small_objs, medium_objs, large_objs)

    output_data = {
        "image": f"{clean_id}.png",
        "counts": {
            "small": len(small_objs),
            "medium": len(medium_objs),
            "large": len(large_objs)
        },
        "relations_small_medium_count": len(rels_sm),
        "relations_small_medium": rels_sm,
        "relations_small_large_count": len(rels_sl),
        "relations_small_large": rels_sl
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{clean_id}_small_yolo.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return output_data


def run_batch():
    logger.info("=== BẮT ĐẦU CHẠY BATCH SMALL x YOLO (30 ÁNH VALIDATION) ===")
    small_files = sorted([f for f in os.listdir(SMALL_DIR) if f.endswith("_small.json")])
    image_ids = [f.replace("_small.json", "") for f in small_files]

    total_images = len(image_ids)
    sm_case_counts = {}
    sl_case_counts = {}
    total_sm = 0
    total_sl = 0

    for img_id in image_ids:
        res = process_image(img_id)
        r_sm = res["relations_small_medium"]
        r_sl = res["relations_small_large"]

        total_sm += len(r_sm)
        total_sl += len(r_sl)

        for r in r_sm:
            c = r["case"]
            sm_case_counts[c] = sm_case_counts.get(c, 0) + 1

        for r in r_sl:
            c = r["case"]
            sl_case_counts[c] = sl_case_counts.get(c, 0) + 1

    print("\n" + "=" * 70)
    print("=== THỐNG KÊ TỔNG KẾT BATCH PROCESSING (SMALL x YOLO) ===")
    print("=" * 70)
    print(f"  - Tổng số ảnh đã xử lý: {total_images}")
    print("-" * 70)
    print(f"  1. MA TRẬN SMALL x MEDIUM ({total_sm} quan hệ):")
    for case_name, count in sm_case_counts.items():
        pct = (count / total_sm * 100) if total_sm > 0 else 0.0
        print(f"  * {case_name:<25}: {count:>4} pairs ({pct:>6.2f}%)")

    print("-" * 70)
    print(f"  2. MA TRẬN SMALL x LARGE ({total_sl} quan hệ):")
    for case_name, count in sl_case_counts.items():
        pct = (count / total_sl * 100) if total_sl > 0 else 0.0
        print(f"  * {case_name:<25}: {count:>4} pairs ({pct:>6.2f}%)")

    print("=" * 70)
    print(f"File kết quả JSON lưu tại: {RESULTS_DIR}\n")


if __name__ == "__main__":
    run_batch()
