import os
import sys
import json
import logging
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from combine.full_pipeline.graph_builder import build_element_graph
from combine.full_pipeline.hierarchy_resolver import resolve_hierarchy

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("full_pipeline.run_pipeline")

VAL_IMAGES_DIR = os.path.join(BASE_DIR, "yolo_pixelweb_dataset", "images", "val")
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, "data", "yolo_pixelweb_dataset", "images", "val")

RESULTS_DIR = os.path.join(BASE_DIR, "combine", "full_pipeline", "results", "json")


def process_image(img_id: str) -> Dict[str, Any]:
    clean_id = img_id.replace('.png', '')

    all_nodes, all_edges = build_element_graph(clean_id)
    elements = resolve_hierarchy(all_nodes, all_edges)

    output_data = {
        "image": f"{clean_id}.png",
        "total_elements": len(elements),
        "total_raw_nodes": len(all_nodes),
        "total_raw_edges": len(all_edges),
        "elements": elements
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{clean_id}_full_hierarchy.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return output_data


def run_batch():
    logger.info("=== BẮT ĐẦU CHẠY BATCH FULL PIPELINE HIERARCHY BUILDER (30 ÁNH VALIDATION) ===")
    files = sorted([f for f in os.listdir(VAL_IMAGES_DIR) if f.endswith('.png')])[:30]
    image_ids = [f.replace('.png', '') for f in files]

    total_images = len(image_ids)
    total_elements = 0
    total_roots = 0
    total_with_parent = 0
    total_gaps = 0
    source_counts: Dict[str, int] = {}

    for img_id in image_ids:
        res = process_image(img_id)
        elems = res["elements"]
        total_elements += len(elems)

        for el in elems:
            src = el["source"]
            source_counts[src] = source_counts.get(src, 0) + 1

            if el["primary_parent"] is None:
                total_roots += 1
            else:
                total_with_parent += 1

            if el["hierarchy_gap"]:
                total_gaps += 1

    print("\n" + "=" * 70)
    print("=== THỐNG KÊ TỔNG KẾT FULL PIPELINE HIERARCHY BUILDER ===")
    print("=" * 70)
    print(f"  - Tổng số ảnh đã xử lý          : {total_images}")
    print(f"  - Tổng số UI Elements (All 5 src): {total_elements}")
    print(f"  - Số Root Elements (No parent)  : {total_roots} ({(total_roots/total_elements*100):.2f}%)")
    print(f"  - Số Child Elements (Has parent): {total_with_parent} ({(total_with_parent/total_elements*100):.2f}%)")
    print(f"  - Số Elements bị Hierarchy Gap  : {total_gaps} ({(total_gaps/total_elements*100):.2f}%)")
    print("-" * 70)
    print("  PHÂN PHỐI UI ELEMENTS THEO NGUỒN (SOURCE):")
    for src_name, count in source_counts.items():
        pct = (count / total_elements * 100) if total_elements > 0 else 0.0
        print(f"  * {src_name:<15}: {count:>4} elements ({pct:>6.2f}%)")

    print("=" * 70)
    print(f"File kết quả JSON lưu tại: {RESULTS_DIR}\n")


if __name__ == "__main__":
    run_batch()
