import sys
import os
import numpy as np
from typing import List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from combine._core.relation_case import RelationCase
from combine._core.matrix import compute_all_matrices
from combine._core.thresholds import (
    IOU_DUPLICATE_THRESH,
    AREA_RATIO_DUPLICATE_THRESH,
    CONTAINMENT_NESTED_THRESH,
    CONTAINMENT_PARTIAL_THRESH
)


def classify_large_x_large(
    large_objs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Phân loại tự so khớp giữa các box trong cùng tập Large YOLO (A = Large, B = Large).
    Loại trừ tự so sánh i == j và tránh double counting (chỉ xét i < j).
    """
    num_objs = len(large_objs)
    if num_objs < 2:
        return []

    boxes = np.array([obj['bbox'] for obj in large_objs], dtype=np.float64)

    matrix_dict = compute_all_matrices(boxes, boxes)
    iou_mat = matrix_dict['iou']
    cont_a_in_b_mat = matrix_dict['cont_a_in_b']
    cont_b_in_a_mat = matrix_dict['cont_b_in_a']
    ratio_mat = matrix_dict['ratio']

    relations = []

    # Chỉ lặp qua cặp (i, j) với i < j để tránh trùng lặp cặp (i, j) và (j, i)
    for i in range(num_objs):
        for j in range(i + 1, num_objs):
            iou = float(iou_mat[i, j])
            cont_a_in_b = float(cont_a_in_b_mat[i, j])
            cont_b_in_a = float(cont_b_in_a_mat[i, j])
            ratio = float(ratio_mat[i, j])

            area_a = large_objs[i]['area']
            area_b = large_objs[j]['area']

            if iou >= IOU_DUPLICATE_THRESH and ratio >= AREA_RATIO_DUPLICATE_THRESH:
                assigned_case = RelationCase.DUPLICATE
                depth_hint = "same_level"
            elif cont_a_in_b >= CONTAINMENT_NESTED_THRESH and area_a < area_b:
                assigned_case = RelationCase.A_NESTED_IN_B
                depth_hint = "a_inside_b"
            elif cont_b_in_a >= CONTAINMENT_NESTED_THRESH and area_b < area_a:
                assigned_case = RelationCase.B_NESTED_IN_A
                depth_hint = "b_inside_a"
            elif cont_a_in_b >= CONTAINMENT_PARTIAL_THRESH or cont_b_in_a >= CONTAINMENT_PARTIAL_THRESH:
                assigned_case = RelationCase.PARTIAL_OVERLAP
                depth_hint = "overlapping"
            else:
                assigned_case = RelationCase.STANDALONE
                depth_hint = "separate"

            if assigned_case != RelationCase.STANDALONE:
                relations.append({
                    "a_id": large_objs[i]['id'],
                    "a_source": "large",
                    "b_id": large_objs[j]['id'],
                    "b_source": "large",
                    "case": assigned_case.value,
                    "depth_hint": depth_hint,
                    "metrics": {
                        "iou": round(iou, 3),
                        "cont_a_in_b": round(cont_a_in_b, 3),
                        "cont_b_in_a": round(cont_b_in_a, 3),
                        "area_ratio": round(ratio, 3)
                    }
                })

    return relations
