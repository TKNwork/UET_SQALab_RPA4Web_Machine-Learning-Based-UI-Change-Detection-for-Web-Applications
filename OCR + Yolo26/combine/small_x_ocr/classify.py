import numpy as np
from typing import List, Dict, Any, Optional


def classify_small_boxes(
    small_objs: List[Dict[str, Any]],
    ocr_objs: List[Dict[str, Any]],
    matrix_dict: Dict[str, np.ndarray]
) -> List[Dict[str, Any]]:
    """
    Phân loại các box Small YOLO theo 4 case dựa trên ma trận tính toán:
    1. duplicate_ocr          : iou >= 0.85 và area_ratio >= 0.85
    2. nested_in_ocr           : cont_small_in_ocr >= 0.95 và area_ratio < 0.85
    3. partial_overlap_review  : 0.5 <= cont_small_in_ocr < 0.95
    4. standalone_pending_yolo_check : cont_small_in_ocr < 0.5 với mọi OCR box
    """
    results = []
    iou_mat = matrix_dict['iou']
    cont_a_in_b_mat = matrix_dict['cont_a_in_b']
    cont_b_in_a_mat = matrix_dict['cont_b_in_a']
    ratio_mat = matrix_dict['ratio']

    num_ocr = len(ocr_objs)

    for i, s_obj in enumerate(small_objs):
        small_id = s_obj['id']
        bbox = s_obj['bbox']
        confidence = s_obj['confidence']

        if num_ocr == 0:
            results.append({
                "small_id": small_id,
                "bbox": bbox,
                "confidence": confidence,
                "case": "standalone_pending_yolo_check",
                "primary_ocr_match": None,
                "metrics": None
            })
            continue

        best_j = int(np.argmax(cont_a_in_b_mat[i, :]))
        best_cont = float(cont_a_in_b_mat[i, best_j])
        best_iou = float(iou_mat[i, best_j])
        best_cont_b_in_a = float(cont_b_in_a_mat[i, best_j])
        best_ratio = float(ratio_mat[i, best_j])
        best_ocr_text = ocr_objs[best_j]['label']

        # Đánh giá thứ tự ưu tiên bằng if / elif
        if best_iou >= 0.85 and best_ratio >= 0.85:
            assigned_case = "duplicate_ocr"
        elif best_cont >= 0.95 and best_ratio < 0.85:
            assigned_case = "nested_in_ocr"
        elif 0.5 <= best_cont < 0.95:
            assigned_case = "partial_overlap_review"
        else:
            assigned_case = "standalone_pending_yolo_check"

        if assigned_case == "standalone_pending_yolo_check":
            primary_ocr_match = None
            metrics = None
        else:
            primary_ocr_match = best_ocr_text
            metrics = {
                "iou": round(best_iou, 3),
                "cont_small_in_ocr": round(best_cont, 3),
                "cont_ocr_in_small": round(best_cont_b_in_a, 3),
                "area_ratio": round(best_ratio, 3)
            }

        results.append({
            "small_id": small_id,
            "bbox": bbox,
            "confidence": confidence,
            "case": assigned_case,
            "primary_ocr_match": primary_ocr_match,
            "metrics": metrics
        })

    return results
