import sys
import os
import numpy as np
from typing import List, Dict, Any, Tuple

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


def _classify_pair(
    objs_a: List[Dict[str, Any]],
    objs_b: List[Dict[str, Any]],
    source_b_name: str
) -> List[Dict[str, Any]]:
    """Phân loại quan hệ không gian giữa Small YOLO (A) và YOLO Medium/Large (B)"""
    if not objs_a or not objs_b:
        return []

    boxes_a = np.array([obj['bbox'] for obj in objs_a], dtype=np.float64)
    boxes_b = np.array([obj['bbox'] for obj in objs_b], dtype=np.float64)

    matrix_dict = compute_all_matrices(boxes_a, boxes_b)
    iou_mat = matrix_dict['iou']
    cont_a_in_b_mat = matrix_dict['cont_a_in_b']
    cont_b_in_a_mat = matrix_dict['cont_b_in_a']
    ratio_mat = matrix_dict['ratio']

    relations = []
    num_a = len(objs_a)
    num_b = len(objs_b)

    for i in range(num_a):
        for j in range(num_b):
            iou = float(iou_mat[i, j])
            cont_a_in_b = float(cont_a_in_b_mat[i, j])
            cont_b_in_a = float(cont_b_in_a_mat[i, j])
            ratio = float(ratio_mat[i, j])

            area_a = objs_a[i]['area']
            area_b = objs_b[j]['area']

            if iou >= IOU_DUPLICATE_THRESH and ratio >= AREA_RATIO_DUPLICATE_THRESH:
                assigned_case = RelationCase.DUPLICATE
            elif cont_a_in_b >= CONTAINMENT_NESTED_THRESH and area_a < area_b:
                assigned_case = RelationCase.A_NESTED_IN_B
            elif cont_b_in_a >= CONTAINMENT_NESTED_THRESH and area_b < area_a:
                assigned_case = RelationCase.B_NESTED_IN_A
            elif cont_a_in_b >= CONTAINMENT_PARTIAL_THRESH or cont_b_in_a >= CONTAINMENT_PARTIAL_THRESH:
                assigned_case = RelationCase.PARTIAL_OVERLAP
            else:
                assigned_case = RelationCase.STANDALONE

            if assigned_case != RelationCase.STANDALONE:
                relations.append({
                    "a_id": objs_a[i]['id'],
                    "a_source": "small",
                    "b_id": objs_b[j]['id'],
                    "b_source": source_b_name,
                    "case": assigned_case.value,
                    "metrics": {
                        "iou": round(iou, 3),
                        "cont_a_in_b": round(cont_a_in_b, 3),
                        "cont_b_in_a": round(cont_b_in_a, 3),
                        "area_ratio": round(ratio, 3)
                    }
                })

    return relations


def classify_small_x_yolo(
    small_objs: List[Dict[str, Any]],
    medium_objs: List[Dict[str, Any]],
    large_objs: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Tính 2 ma trận độc lập:
    1. Small x Medium
    2. Small x Large
    """
    rels_small_medium = _classify_pair(small_objs, medium_objs, "medium")
    rels_small_large = _classify_pair(small_objs, large_objs, "large")

    return rels_small_medium, rels_small_large
