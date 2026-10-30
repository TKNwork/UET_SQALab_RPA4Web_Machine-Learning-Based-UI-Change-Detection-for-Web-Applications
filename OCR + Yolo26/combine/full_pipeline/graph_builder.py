import os
import sys
import json
import logging
from typing import List, Dict, Any, Set, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from combine._core.schema import load_ocr, load_small, load_medium, load_large, load_cv

logger = logging.getLogger("full_pipeline.graph_builder")


def build_element_graph(img_id: str) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Nạp tất cả các node từ 5 nguồn và thu thập toàn bộ các cạnh quan hệ từ tất cả các module cặp nguồn.
    """
    clean_id = img_id.replace('.png', '')

    # 1. Nạp tất cả các node từ 5 nguồn
    ocr_objs = load_ocr(clean_id, BASE_DIR)
    small_objs = load_small(clean_id, BASE_DIR)
    med_objs = load_medium(clean_id, BASE_DIR)
    large_objs = load_large(clean_id, BASE_DIR)
    cv_objs = load_cv(clean_id, BASE_DIR)

    all_nodes: Dict[str, Dict[str, Any]] = {}
    for obj in ocr_objs + small_objs + med_objs + large_objs + cv_objs:
        all_nodes[obj['id']] = obj

    # 2. Đọc tất cả các file JSON quan hệ đã xuất từ các module
    all_edges: List[Dict[str, Any]] = []

    rel_paths = [
        os.path.join(BASE_DIR, 'combine', 'small_x_cv', 'results', 'json', f"{clean_id}_small_cv.json"),
        os.path.join(BASE_DIR, 'combine', 'small_x_yolo', 'results', 'json', f"{clean_id}_small_yolo.json"),
        os.path.join(BASE_DIR, 'combine', 'medium_x_ocr', 'results', 'json', f"{clean_id}_medium_ocr.json"),
        os.path.join(BASE_DIR, 'combine', 'medium_x_cv', 'results', 'json', f"{clean_id}_medium_cv.json"),
        os.path.join(BASE_DIR, 'combine', 'medium_x_large', 'results', 'json', f"{clean_id}_medium_large.json"),
        os.path.join(BASE_DIR, 'combine', 'cv_x_large', 'results', 'json', f"{clean_id}_cv_large.json"),
        os.path.join(BASE_DIR, 'combine', 'ocr_x_large', 'results', 'json', f"{clean_id}_ocr_large.json"),
        os.path.join(BASE_DIR, 'combine', 'large_x_large', 'results', 'json', f"{clean_id}_large_large.json"),
        os.path.join(BASE_DIR, 'combine', 'small_x_ocr', 'results', 'json', f"{clean_id}_small_ocr_match.json")
    ]

    for p in rel_paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Xử lý các format JSON khác nhau
            if 'relations' in data:
                all_edges.extend(data['relations'])
            if 'relations_small_medium' in data:
                all_edges.extend(data['relations_small_medium'])
            if 'relations_small_large' in data:
                all_edges.extend(data['relations_small_large'])
            if 'matches' in data: # Từ small_x_ocr demo
                for m in data['matches']:
                    if m.get('case') in ('duplicate_ocr', 'nested_in_ocr', 'partial_overlap_review') and m.get('primary_ocr_match'):
                        all_edges.append({
                            "a_id": m['small_id'],
                            "a_source": "small",
                            "b_id": "ocr_matched", # Sẽ được định hướng nếu trùng
                            "b_source": "ocr",
                            "case": "duplicate" if m['case'] == 'duplicate_ocr' else "a_nested_in_b",
                            "metrics": m.get('metrics')
                        })

    return all_nodes, all_edges
