import os
import json
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_and_normalize_box(bbox: List[float]) -> List[float]:
    """Quy đổi và kiểm tra box xem có bị suy biến (x2 <= x1 hoặc y2 <= y1) không."""
    x1, y1, x2, y2 = [float(c) for c in bbox]
    if x2 <= x1 or y2 <= y1:
        logger.warning(f"Phát hiện bounding box suy biến: [{x1}, {y1}, {x2}, {y2}]")
    return [x1, y1, x2, y2]


def normalize_yolo_small(json_path: str) -> List[Dict[str, Any]]:
    """
    Đọc file JSON kết quả YOLO small (`yolo/results/small/<img_id>_small.json`)
    và chuẩn hóa về schema chung.
    """
    if not os.path.exists(json_path):
        logger.warning(f"File YOLO small không tồn tại: {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    detections = data.get("detections", [])
    normalized = []

    for idx, item in enumerate(detections):
        raw_bbox = item.get("bbox", [0, 0, 0, 0])
        bbox = validate_and_normalize_box(raw_bbox)
        x1, y1, x2, y2 = bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        conf = float(item.get("confidence", 0.0))
        label = str(item.get("class", "ui_element"))

        normalized.append({
            "id": f"small_{idx}",
            "source": "yolo",
            "bbox": bbox,
            "confidence": conf,
            "area": round(area, 4),
            "label": label
        })

    return normalized


def normalize_ocr(json_path: str) -> List[Dict[str, Any]]:
    """
    Đọc file JSON kết quả OCR (`ocr/results/<img_id>.json`)
    và chuẩn hóa về schema chung.
    """
    if not os.path.exists(json_path):
        logger.warning(f"File OCR không tồn tại: {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    paragraphs = data.get("paragraphs", [])
    normalized = []

    for idx, item in enumerate(paragraphs):
        raw_bbox = item.get("bbox", [0, 0, 0, 0])
        bbox = validate_and_normalize_box(raw_bbox)
        x1, y1, x2, y2 = bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        conf = float(item.get("confidence", 0.0))
        text = str(item.get("text", ""))

        normalized.append({
            "id": f"ocr_{idx}",
            "source": "ocr",
            "bbox": bbox,
            "confidence": conf,
            "area": round(area, 4),
            "label": text
        })

    return normalized
