import os
import json
import logging
from typing import List, Dict, Any, Optional

from combine._core.geometry import to_xyxy, compute_area, validate_box

logger = logging.getLogger("combine._core.schema")

# Đường dẫn mặc định thư mục gốc của project
DEFAULT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_base_dir(base_dir: Optional[str] = None) -> str:
    return base_dir if base_dir else DEFAULT_BASE_DIR


def load_ocr(img_id: str, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Nạp và chuẩn hóa dữ liệu từ ocr/results/<img_id>.json
    """
    b_dir = _get_base_dir(base_dir)
    clean_id = img_id.replace('.png', '')
    json_path = os.path.join(b_dir, 'ocr', 'results', f"{clean_id}.json")

    if not os.path.exists(json_path):
        logger.warning(f"File OCR không tồn tại: {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    paragraphs = data.get("paragraphs", [])
    normalized = []

    for idx, item in enumerate(paragraphs):
        raw_bbox = item.get("bbox", [0, 0, 0, 0])
        bbox = [float(c) for c in raw_bbox]

        if not validate_box(bbox):
            continue

        area = compute_area(bbox)
        conf = float(item.get("confidence", 0.0))
        text = str(item.get("text", ""))

        normalized.append({
            "id": f"ocr_{idx}",
            "source": "ocr",
            "bbox": bbox,
            "confidence": conf,
            "area": area,
            "label": text,
            "text": text
        })

    return normalized


def load_small(img_id: str, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Nạp và chuẩn hóa dữ liệu từ yolo/results/small/<img_id>_small.json
    """
    b_dir = _get_base_dir(base_dir)
    clean_id = img_id.replace('.png', '').replace('_small', '')
    json_path = os.path.join(b_dir, 'yolo', 'results', 'small', f"{clean_id}_small.json")

    if not os.path.exists(json_path):
        logger.warning(f"File YOLO Small không tồn tại: {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    detections = data.get("detections", [])
    normalized = []

    for idx, item in enumerate(detections):
        raw_bbox = item.get("bbox", [0, 0, 0, 0])
        bbox = [float(c) for c in raw_bbox]

        if not validate_box(bbox):
            continue

        area = compute_area(bbox)
        conf = float(item.get("confidence", 0.0))
        cls_name = str(item.get("class", "ui_element"))

        normalized.append({
            "id": f"small_{idx}",
            "source": "small",
            "bbox": bbox,
            "confidence": conf,
            "area": area,
            "label": cls_name,
            "text": None
        })

    return normalized


def load_medium(img_id: str, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Nạp và chuẩn hóa dữ liệu từ yolo/results/medium/<img_id>_medium.json
    """
    b_dir = _get_base_dir(base_dir)
    clean_id = img_id.replace('.png', '').replace('_medium', '')
    json_path = os.path.join(b_dir, 'yolo', 'results', 'medium', f"{clean_id}_medium.json")

    if not os.path.exists(json_path):
        logger.warning(f"File YOLO Medium không tồn tại: {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    detections = data.get("detections", [])
    normalized = []

    for idx, item in enumerate(detections):
        raw_bbox = item.get("bbox", [0, 0, 0, 0])
        bbox = [float(c) for c in raw_bbox]

        if not validate_box(bbox):
            continue

        area = compute_area(bbox)
        conf = float(item.get("confidence", 0.0))
        cls_name = str(item.get("class", "ui_element"))

        normalized.append({
            "id": f"medium_{idx}",
            "source": "medium",
            "bbox": bbox,
            "confidence": conf,
            "area": area,
            "label": cls_name,
            "text": None
        })

    return normalized


def load_large(img_id: str, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Nạp và chuẩn hóa dữ liệu từ yolo/results/large/<img_id>_large.json
    """
    b_dir = _get_base_dir(base_dir)
    clean_id = img_id.replace('.png', '').replace('_large', '')
    json_path = os.path.join(b_dir, 'yolo', 'results', 'large', f"{clean_id}_large.json")

    if not os.path.exists(json_path):
        logger.warning(f"File YOLO Large không tồn tại: {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    detections = data.get("detections", [])
    normalized = []

    for idx, item in enumerate(detections):
        raw_bbox = item.get("bbox", [0, 0, 0, 0])
        bbox = [float(c) for c in raw_bbox]

        if not validate_box(bbox):
            continue

        area = compute_area(bbox)
        conf = float(item.get("confidence", 0.0))
        cls_name = str(item.get("class", "ui_element"))

        normalized.append({
            "id": f"large_{idx}",
            "source": "large",
            "bbox": bbox,
            "confidence": conf,
            "area": area,
            "label": cls_name,
            "text": None
        })

    return normalized


def load_cv(img_id: str, base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Nạp và chuẩn hóa dữ liệu từ cv/results/json/<img_id>_cv.json.
    LƯU Ý: Chuyển đổi BBox của OpenCV [x, y, w, h] sang [x1, y1, x2, y2] = [x, y, x + w, y + h].
    """
    b_dir = _get_base_dir(base_dir)
    clean_id = img_id.replace('.png', '').replace('_cv', '')
    json_path = os.path.join(b_dir, 'cv', 'results', 'json', f"{clean_id}_cv.json")

    if not os.path.exists(json_path):
        logger.warning(f"File CV không tồn tại: {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    detections = data.get("detections", [])
    normalized = []

    for idx, item in enumerate(detections):
        raw_bbox = item.get("bbox", [0, 0, 0, 0])
        # Quy đổi [x, y, w, h] -> [x1, y1, x2, y2]
        bbox = to_xyxy(raw_bbox, fmt='xywh')

        if not validate_box(bbox):
            continue

        area = compute_area(bbox)
        conf = float(item.get("confidence", 1.0))
        label = str(item.get("label", "rectangle_box"))

        normalized.append({
            "id": f"cv_{idx}",
            "source": "cv",
            "bbox": bbox,
            "confidence": conf,
            "area": area,
            "label": label,
            "class_label": label,
            "text": None
        })

    return normalized


def normalize_ocr_nodes(paragraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chuẩn hóa danh sách paragraph từ run_ocr() trong bộ nhớ.
    """
    normalized = []
    for idx, item in enumerate(paragraphs):
        raw_bbox = item.get("bbox", [0, 0, 0, 0])
        bbox = [float(c) for c in raw_bbox]
        if not validate_box(bbox):
            continue
        area = compute_area(bbox)
        conf = float(item.get("confidence", 0.0))
        text = str(item.get("text", "")).strip()

        normalized.append({
            "id": f"ocr_{idx}",
            "source": "ocr",
            "class_label": "text",
            "label": "text",
            "bbox": bbox,
            "confidence": conf,
            "area": area,
            "text": text if text else None
        })
    return normalized


def normalize_yolo_nodes(detections: List[Dict[str, Any]], size_cat: str) -> List[Dict[str, Any]]:
    """
    Chuẩn hóa danh sách detection YOLO (small/medium/large) từ run_yolo() trong bộ nhớ.
    """
    normalized = []
    for idx, item in enumerate(detections):
        raw_bbox = item.get("bbox", [0, 0, 0, 0])
        bbox = [float(c) for c in raw_bbox]
        if not validate_box(bbox):
            continue
        area = compute_area(bbox)
        conf = float(item.get("confidence", 0.0))
        cls_name = str(item.get("class", "ui_element"))

        normalized.append({
            "id": f"{size_cat}_{idx}",
            "source": size_cat,
            "class_label": cls_name,
            "label": cls_name,
            "bbox": bbox,
            "confidence": conf,
            "area": area,
            "text": None
        })
    return normalized


def normalize_cv_nodes(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chuẩn hóa danh sách detection từ run_cv() trong bộ nhớ.
    Quy đổi [x, y, w, h] sang [x1, y1, x2, y2].
    """
    normalized = []
    for idx, item in enumerate(detections):
        raw_bbox = item.get("bbox", [0, 0, 0, 0])
        bbox = to_xyxy(raw_bbox, fmt='xywh')
        if not validate_box(bbox):
            continue
        area = compute_area(bbox)
        conf = float(item.get("confidence", 1.0))
        label = str(item.get("label", "rectangle_box"))

        normalized.append({
            "id": f"cv_{idx}",
            "source": "cv",
            "class_label": label,
            "label": label,
            "bbox": bbox,
            "confidence": conf,
            "area": area,
            "text": None
        })
    return normalized

