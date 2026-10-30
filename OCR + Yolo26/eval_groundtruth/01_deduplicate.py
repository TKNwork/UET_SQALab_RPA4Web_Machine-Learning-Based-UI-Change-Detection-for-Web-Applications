import os
import json
import numpy as np
from typing import List, Dict, Any

from combine._core.matrix import compute_all_matrices

# Trọng số ưu tiên nguồn cho WBF (YOLO được gán trọng số cao hơn để định hướng viền & confidence)
SOURCE_WEIGHTS = {
    'small': 1.2,
    'medium': 1.2,
    'large': 1.2,
    'ocr': 1.0,
    'cv': 1.0
}


def can_fuse_elements(elem_a: Dict[str, Any], elem_b: Dict[str, Any], iou: float, iou_thresh: float = 0.85) -> bool:
    """
    Chỉ cho phép WBF hợp nhất nếu:
    1. IoU >= iou_thresh (mặc định 0.85)
    VÀ thuộc một trong các case quy định:
    - Case 1: YOLO (small / medium / large) duplicate OCR
    - Case 2: YOLO (small / medium / large) duplicate CV_rectangle (label == 'rectangle_box')
    - Case 3: YOLO duplicate YOLO (cùng hoặc khác size)
    """
    if iou < iou_thresh:
        return False

    src_a = elem_a.get('source', '')
    src_b = elem_b.get('source', '')

    is_yolo_a = src_a in ('small', 'medium', 'large')
    is_yolo_b = src_b in ('small', 'medium', 'large')

    is_ocr_a = (src_a == 'ocr')
    is_ocr_b = (src_b == 'ocr')

    is_cv_rect_a = (src_a == 'cv' and elem_a.get('class_label') == 'rectangle_box')
    is_cv_rect_b = (src_b == 'cv' and elem_b.get('class_label') == 'rectangle_box')

    # 1. YOLO x YOLO
    if is_yolo_a and is_yolo_b:
        return True

    # 2. YOLO x OCR
    if (is_yolo_a and is_ocr_b) or (is_yolo_b and is_ocr_a):
        return True

    # 3. YOLO x CV Rectangle
    if (is_yolo_a and is_cv_rect_b) or (is_yolo_b and is_cv_rect_a):
        return True

    return False


def apply_wbf_deduplication(elements: List[Dict[str, Any]], iou_thresh: float = 0.85) -> List[Dict[str, Any]]:
    """
    Áp dụng thuật toán Weighted Boxes Fusion (WBF) cải tiến:
    - Chỉ gộp các case: YOLO duplicate OCR, YOLO duplicate CV_rectangle, và YOLO duplicate YOLO.
    - Điểm confidence và tọa độ BBox được tính bằng TRUNG BÌNH TRỌNG SỐ giữa các nguồn.
    """
    if not elements:
        return []

    # Sắp xếp elements giảm dần theo confidence
    sorted_elems = sorted(elements, key=lambda e: e.get('confidence', 0.0), reverse=True)
    fused_elements = []

    visited = set()

    for i, elem_i in enumerate(sorted_elems):
        if i in visited:
            continue

        box_i = elem_i['bbox']
        cluster = [elem_i]
        visited.add(i)

        for j in range(i + 1, len(sorted_elems)):
            if j in visited:
                continue

            elem_j = sorted_elems[j]
            box_j = elem_j['bbox']

            # Tính IoU giữa box_i và box_j
            boxes_a = np.array([box_i], dtype=np.float64)
            boxes_b = np.array([box_j], dtype=np.float64)
            mat = compute_all_matrices(boxes_a, boxes_b)
            iou = mat['iou'][0, 0]

            if can_fuse_elements(elem_i, elem_j, iou, iou_thresh):
                cluster.append(elem_j)
                visited.add(j)

        # Tính TRUNG BÌNH TRỌNG SỐ giữa các nguồn cho điểm confidence và tọa độ BBox
        weights = [SOURCE_WEIGHTS.get(e.get('source', ''), 1.0) * float(e.get('confidence', 0.5)) for e in cluster]
        total_w = sum(weights)

        # 1. Hợp nhất Tọa độ BBox bằng Trung bình Trọng số
        w_x1 = sum(e['bbox'][0] * w for e, w in zip(cluster, weights)) / total_w
        w_y1 = sum(e['bbox'][1] * w for e, w in zip(cluster, weights)) / total_w
        w_x2 = sum(e['bbox'][2] * w for e, w in zip(cluster, weights)) / total_w
        w_y2 = sum(e['bbox'][3] * w for e, w in zip(cluster, weights)) / total_w

        fused_bbox = [round(w_x1, 2), round(w_y1, 2), round(w_x2, 2), round(w_y2, 2)]

        # 2. Hợp nhất điểm Confidence bằng Trung bình Trọng số giữa các nguồn
        w_conf = sum(float(e.get('confidence', 0.5)) * w for e, w in zip(cluster, weights)) / total_w

        # Chọn nhãn ưu tiên theo phần tử YOLO hoặc phần tử có điểm cao nhất
        yolo_in_cluster = [e for e in cluster if e.get('source') in ('small', 'medium', 'large')]
        primary_elem = yolo_in_cluster[0] if yolo_in_cluster else cluster[0]

        # Lấy mảng các nguồn duy nhất đã gộp trong cluster
        sources_list = []
        for e in cluster:
            s = e.get('source', '')
            if s and s not in sources_list:
                sources_list.append(s)

        # Trích xuất text từ OCR nếu có
        cluster_text = next((e['text'] for e in cluster if e.get('text')), None)

        class_lbl = primary_elem.get('class_label', primary_elem.get('label', 'ui_element'))

        area_val = round(max(0.0, fused_bbox[2] - fused_bbox[0]) * max(0.0, fused_bbox[3] - fused_bbox[1]), 2)

        fused_elements.append({
            "id": f"fused_{len(fused_elements)}",
            "source": primary_elem['source'],
            "sources": sources_list,
            "class_label": class_lbl,
            "label": class_lbl,
            "bbox": fused_bbox,
            "confidence": round(w_conf, 4),
            "cluster_size": len(cluster),
            "area": area_val,
            "text": cluster_text
        })

    return fused_elements

