import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_RESULTS_DIR = os.path.join(BASE_DIR, 'yolo', 'results')

SMALL_DIR = os.path.join(YOLO_RESULTS_DIR, 'small')
MEDIUM_DIR = os.path.join(YOLO_RESULTS_DIR, 'medium')
LARGE_DIR = os.path.join(YOLO_RESULTS_DIR, 'large')

SIZE_THRESHOLDS = {
    'small': 0.10,
    'medium': 0.50,
    'large': 0.85
}


def split_yolo_outputs():
    os.makedirs(SMALL_DIR, exist_ok=True)
    os.makedirs(MEDIUM_DIR, exist_ok=True)
    os.makedirs(LARGE_DIR, exist_ok=True)

    json_files = sorted([
        f for f in os.listdir(YOLO_RESULTS_DIR)
        if f.endswith('.json') and not f.startswith('.')
    ])

    print(f"-> Đã tìm thấy {len(json_files)} file JSON trong '{YOLO_RESULTS_DIR}'")

    for file_name in json_files:
        filepath = os.path.join(YOLO_RESULTS_DIR, file_name)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_name = data.get("image", file_name.replace('.json', '.png'))
        detections = data.get("detections", [])

        small_dets = [d for d in detections if d.get("size_category") == "small"]
        medium_dets = [d for d in detections if d.get("size_category") == "medium"]
        large_dets = [d for d in detections if d.get("size_category") == "large"]

        base_name = os.path.splitext(file_name)[0]

        # 1. Ghi file small
        small_content = {
            "image": image_name,
            "size_category": "small",
            "confidence_threshold": SIZE_THRESHOLDS["small"],
            "detections_count": len(small_dets),
            "detections": small_dets
        }
        with open(os.path.join(SMALL_DIR, f"{base_name}_small.json"), "w", encoding="utf-8") as f:
            json.dump(small_content, f, ensure_ascii=False, indent=2)

        # 2. Ghi file medium
        medium_content = {
            "image": image_name,
            "size_category": "medium",
            "confidence_threshold": SIZE_THRESHOLDS["medium"],
            "detections_count": len(medium_dets),
            "detections": medium_dets
        }
        with open(os.path.join(MEDIUM_DIR, f"{base_name}_medium.json"), "w", encoding="utf-8") as f:
            json.dump(medium_content, f, ensure_ascii=False, indent=2)

        # 3. Ghi file large
        large_content = {
            "image": image_name,
            "size_category": "large",
            "confidence_threshold": SIZE_THRESHOLDS["large"],
            "detections_count": len(large_dets),
            "detections": large_dets
        }
        with open(os.path.join(LARGE_DIR, f"{base_name}_large.json"), "w", encoding="utf-8") as f:
            json.dump(large_content, f, ensure_ascii=False, indent=2)

        print(f"  + [{base_name}] Split successfully -> Small: {len(small_dets)}, Med: {len(medium_dets)}, Large: {len(large_dets)}")

    print("\n=== HOÀN THÀNH TÁCH OUTPUT YOLO SANG 3 THƯ MỤC SIZE COCO ===")
    print(f"  - Small : {SMALL_DIR}")
    print(f"  - Medium: {MEDIUM_DIR}")
    print(f"  - Large : {LARGE_DIR}")


if __name__ == "__main__":
    split_yolo_outputs()
