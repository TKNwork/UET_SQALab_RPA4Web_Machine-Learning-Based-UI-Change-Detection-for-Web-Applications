from typing import List, Dict, Any, Tuple


def format_coco_dataset(
    images_gt: List[Tuple[str, int, int, List[Dict[str, Any]]]],
    images_dt: List[Tuple[str, List[Dict[str, Any]]]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Quy đổi tập Ground Truth và tập Predictions về định dạng COCO API tiêu chuẩn.
    BBox trong COCO có dạng [x_min, y_min, width, height].
    """
    coco_gt = {
        "images": [],
        "categories": [{"id": 0, "name": "ui_element"}],
        "annotations": []
    }

    coco_dt = []
    ann_id_counter = 1

    img_id_map = {}

    for idx, (img_file, img_w, img_h, gt_boxes) in enumerate(images_gt, 1):
        img_id_map[img_file] = idx

        coco_gt["images"].append({
            "id": idx,
            "file_name": img_file,
            "width": img_w,
            "height": img_h
        })

        for gt in gt_boxes:
            x1, y1, x2, y2 = gt['bbox']
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            coco_gt["annotations"].append({
                "id": ann_id_counter,
                "image_id": idx,
                "category_id": 0,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id_counter += 1

    for img_file, dt_boxes in images_dt:
        if img_file not in img_id_map:
            continue
        img_idx = img_id_map[img_file]

        for dt in dt_boxes:
            x1, y1, x2, y2 = dt['bbox']
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            score = float(dt.get('confidence', 0.5))

            coco_dt.append({
                "image_id": img_idx,
                "category_id": 0,
                "bbox": [x1, y1, w, h],
                "score": score
            })

    return coco_gt, coco_dt
