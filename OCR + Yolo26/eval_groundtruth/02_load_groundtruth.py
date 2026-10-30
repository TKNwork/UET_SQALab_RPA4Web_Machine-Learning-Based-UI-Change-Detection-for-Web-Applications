import os
from PIL import Image
from typing import List, Dict, Any, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'yolo_pixelweb_dataset', 'images', 'val')
if not os.path.exists(VAL_IMAGES_DIR):
    VAL_IMAGES_DIR = os.path.join(BASE_DIR, 'data', 'yolo_pixelweb_dataset', 'images', 'val')

GT_LABELS_DIR = os.path.join(BASE_DIR, 'data', 'yolo_pixelweb_dataset', 'labels', 'val')


def get_image_size(img_id: str) -> Tuple[int, int]:
    clean_id = img_id.replace('.png', '')
    img_path = os.path.join(VAL_IMAGES_DIR, f"{clean_id}.png")
    if not os.path.exists(img_path):
        return (800, 400)  # Fallback default (width, height)
    with Image.open(img_path) as img:
        return img.size  # (width, height)


def load_groundtruth_yolo(img_id: str) -> List[Dict[str, Any]]:
    """
    Đọc file Ground Truth nhãn YOLO TXT từ data/yolo_pixelweb_dataset/labels/val/<img_id>.txt
    và quy đổi tọa độ chuẩn hóa sang [x1, y1, x2, y2] tuyệt đối theo px.
    """
    clean_id = img_id.replace('.png', '')
    label_path = os.path.join(GT_LABELS_DIR, f"{clean_id}.txt")

    if not os.path.exists(label_path):
        return []

    img_w, img_h = get_image_size(clean_id)
    gt_boxes = []

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                xc, yc, w, h = [float(v) for v in parts[1:5]]

                x1 = round(max(0.0, (xc - w / 2.0) * img_w), 2)
                y1 = round(max(0.0, (yc - h / 2.0) * img_h), 2)
                x2 = round(min(float(img_w), (xc + w / 2.0) * img_w), 2)
                y2 = round(min(float(img_h), (yc + h / 2.0) * img_h), 2)

                gt_boxes.append({
                    "class_id": class_id,
                    "bbox": [x1, y1, x2, y2],
                    "area": round((x2 - x1) * (y2 - y1), 2)
                })

    return gt_boxes
