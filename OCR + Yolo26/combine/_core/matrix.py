import numpy as np
from typing import Dict, Tuple


def compute_all_matrices(boxes_a: np.ndarray, boxes_b: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Tính ma trận IoU, Containment 2 chiều (cont_a_in_b, cont_b_in_a), và Area Ratio
    bằng phương pháp Vectorized Broadcasting (shape N x M).
    
    :param boxes_a: NumPy array shape (N, 4) chứa [x1, y1, x2, y2]
    :param boxes_b: NumPy array shape (M, 4) chứa [x1, y1, x2, y2]
    :return: Dict chứa 4 ma trận NumPy shape (N, M):
             'iou', 'cont_a_in_b', 'cont_b_in_a', 'ratio'
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        N = boxes_a.shape[0] if boxes_a.ndim == 2 else 0
        M = boxes_b.shape[0] if boxes_b.ndim == 2 else 0
        return {
            'iou': np.zeros((N, M), dtype=np.float64),
            'cont_a_in_b': np.zeros((N, M), dtype=np.float64),
            'cont_b_in_a': np.zeros((N, M), dtype=np.float64),
            'ratio': np.zeros((N, M), dtype=np.float64)
        }

    a = boxes_a[:, None, :]  # (N, 1, 4)
    b = boxes_b[None, :, :]  # (1, M, 4)

    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float64), where=union > 0)

    cont_a_in_b = np.divide(inter, area_a[:, None], out=np.zeros_like(inter, dtype=np.float64), where=area_a[:, None] > 0)
    cont_b_in_a = np.divide(inter, area_b[None, :], out=np.zeros_like(inter, dtype=np.float64), where=area_b[None, :] > 0)

    area_max = np.maximum(area_a[:, None], area_b[None, :])
    area_min = np.minimum(area_a[:, None], area_b[None, :])
    ratio = np.divide(area_min, area_max, out=np.zeros_like(inter, dtype=np.float64), where=area_max > 0)

    return {
        'iou': iou,
        'cont_a_in_b': cont_a_in_b,
        'cont_b_in_a': cont_b_in_a,
        'ratio': ratio
    }
