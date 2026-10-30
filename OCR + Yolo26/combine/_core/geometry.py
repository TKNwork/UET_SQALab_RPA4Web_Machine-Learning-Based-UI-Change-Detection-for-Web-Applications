import logging
from typing import List, Tuple, Union

logger = logging.getLogger("combine._core.geometry")


def to_xyxy(bbox: Union[List[float], Tuple[float, ...]], fmt: str = 'xywh') -> List[float]:
    """
    Chuyển đổi Bounding Box từ định dạng chỉ định sang [x1, y1, x2, y2] tuyệt đối.
    Mặc định fmt='xywh' (OpenCV format: [x, y, width, height]).
    """
    if len(bbox) != 4:
        raise ValueError(f"BBox phải có 4 phần tử, nhận được: {bbox}")

    x, y, w, h = [float(v) for v in bbox]
    if fmt == 'xywh':
        return [x, y, x + w, y + h]
    elif fmt == 'xyxy':
        return [x, y, w, h]  # đã ở dạng xyxy
    else:
        raise ValueError(f"Format không hợp lệ: {fmt}. Chỉ hỗ trợ 'xywh' hoặc 'xyxy'.")


def compute_area(bbox_xyxy: Union[List[float], Tuple[float, ...]]) -> float:
    """
    Tính diện tích của bounding box (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return round(width * height, 4)


def validate_box(bbox_xyxy: Union[List[float], Tuple[float, ...]]) -> bool:
    """
    Kiểm tra box có hợp lệ hay không (x2 > x1 và y2 > y1).
    Log cảnh báo nếu phát hiện box suy biến.
    """
    if len(bbox_xyxy) != 4:
        return False

    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    if x2 <= x1 or y2 <= y1:
        logger.warning(f"Phát hiện Bounding Box suy biến: [{x1}, {y1}, {x2}, {y2}]")
        return False

    return True
