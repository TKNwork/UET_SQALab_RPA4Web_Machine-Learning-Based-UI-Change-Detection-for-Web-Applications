import numpy as np
from typing import Dict, Any, List


def _compute_iou_single(boxA, boxB):
    ax1, ay1, aw, ah = boxA
    bx1, by1, bw, bh = boxB

    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0 else 0.0


def compute_ap_at_iou(gt_anns: List[Dict[str, Any]], dt_anns: List[Dict[str, Any]], iou_thresh: float) -> float:
    """Tính Average Precision (AP) tại một ngưỡng IoU cụ thể"""
    if not gt_anns:
        return 0.0
    if not dt_anns:
        return 0.0

    # Nhóm GT và DT theo image_id
    gt_by_img = {}
    for ann in gt_anns:
        img_id = ann['image_id']
        gt_by_img.setdefault(img_id, []).append({'bbox': ann['bbox'], 'matched': False})

    # Sắp xếp DT theo score giảm dần
    dt_sorted = sorted(dt_anns, key=lambda d: d.get('score', 0.0), reverse=True)

    tp = np.zeros(len(dt_sorted))
    fp = np.zeros(len(dt_sorted))

    num_gt = len(gt_anns)

    for i, dt in enumerate(dt_sorted):
        img_id = dt['image_id']
        dt_box = dt['bbox']

        if img_id not in gt_by_img:
            fp[i] = 1
            continue

        gt_list = gt_by_img[img_id]
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, g in enumerate(gt_list):
            iou = _compute_iou_single(dt_box, g['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0 and not gt_list[best_gt_idx]['matched']:
            tp[i] = 1
            gt_list[best_gt_idx]['matched'] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / float(num_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

    # COCO 101-point interpolation
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i_indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i_indices + 1] - mrec[i_indices]) * mpre[i_indices + 1])
    return float(ap)


def evaluate_map(coco_gt: Dict[str, Any], coco_dt: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Thực thi đánh giá mAP50, mAP75, mAP50-95.
    Thử dùng pycocotools trước, nếu chưa cài đặt sẽ fallback dùng evaluator bằng Python thuần.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt_obj = COCO()
        coco_gt_obj.dataset = coco_gt
        coco_gt_obj.createIndex()

        coco_dt_obj = coco_gt_obj.loadRes(coco_dt)

        coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        return {
            "mAP50_95": float(stats[0]),
            "mAP50": float(stats[1]),
            "mAP75": float(stats[2]),
            "mAP_small": float(stats[3]),
            "mAP_medium": float(stats[4]),
            "mAP_large": float(stats[5])
        }
    except Exception as e:
        print(f"[NOTE] Sử dụng Python COCO Evaluator fallback (Reason: {e})")
        gt_anns = coco_gt.get("annotations", [])
        dt_anns = coco_dt

        iou_thresholds = np.linspace(0.50, 0.95, 10)
        aps = []
        for thresh in iou_thresholds:
            ap = compute_ap_at_iou(gt_anns, dt_anns, float(thresh))
            aps.append(ap)

        ap50 = compute_ap_at_iou(gt_anns, dt_anns, 0.50)
        ap75 = compute_ap_at_iou(gt_anns, dt_anns, 0.75)
        ap_mean = float(np.mean(aps))

        return {
            "mAP50_95": round(ap_mean, 4),
            "mAP50": round(ap50, 4),
            "mAP75": round(ap75, 4),
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0
        }
