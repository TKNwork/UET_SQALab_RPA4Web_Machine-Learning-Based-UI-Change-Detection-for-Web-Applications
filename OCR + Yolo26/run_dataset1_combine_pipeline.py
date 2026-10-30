import os
import sys
import time
import json
import base64
import logging
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger("dataset1_pipeline")

DATASET1_DIR = os.environ.get('RPA_DATASET1') or os.path.join(BASE_DIR, "data", "dataset1")
BASELINE_DIR = os.path.join(DATASET1_DIR, "baseline")
INJECT_DIR = os.path.join(DATASET1_DIR, "inject")

OUTPUT_ROOT_DIR = os.environ.get('RPA_DETECTION_OUTPUT') or os.path.join(BASE_DIR, "output_dataset1")
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

# Modules from project
from ocr.test_gcv_ocr import load_api_key, KEY_FILE, API_ENDPOINT, extract_paragraphs_from_response
from yolo.test_yolo_inference import detect_device, SIZE_THRESHOLDS, predict_and_filter_yolo
from ultralytics import YOLO
from cv.contour_ui_detector import detect_ui_elements

from combine.small_x_cv.classify import classify_small_x_cv
from combine.small_x_yolo.classify import classify_small_x_yolo
from combine.medium_x_ocr.classify import classify_medium_x_ocr
from combine.medium_x_cv.classify import classify_medium_x_cv
from combine.medium_x_large.classify import classify_medium_x_large
from combine.cv_x_large.classify import classify_cv_x_large
from combine.ocr_x_large.classify import classify_ocr_x_large
from combine.large_x_large.classify import classify_large_x_large

from combine.small_x_ocr.matrix import compute_matrix as compute_matrix_small_ocr
from combine.small_x_ocr.classify import classify_small_boxes

from combine.full_pipeline.hierarchy_resolver import resolve_hierarchy
import importlib
dedup_mod = importlib.import_module('eval_groundtruth.01_deduplicate')
apply_wbf_deduplication = dedup_mod.apply_wbf_deduplication


COLOR_SOURCE = {
    'ocr': (0, 255, 0),        # Green
    'small': (255, 0, 0),      # Red
    'medium': (0, 100, 255),   # Blue
    'large': (0, 150, 0),      # Dark Green
    'cv': (147, 112, 219)      # Purple
}


def draw_legend(draw, font):
    legend_box = [10, 10, 270, 140]
    draw.rectangle(legend_box, fill=(0, 0, 0), outline=(255, 255, 255), width=1)
    draw.text((15, 14), "Full Pipeline Hierarchy Legend:", fill=(255, 255, 255), font=font)
    items = [
        ("OCR Text", COLOR_SOURCE['ocr']),
        ("Small YOLO", COLOR_SOURCE['small']),
        ("Medium YOLO", COLOR_SOURCE['medium']),
        ("Large YOLO", COLOR_SOURCE['large']),
        ("CV Contour", COLOR_SOURCE['cv'])
    ]
    y_offset = 32
    for label, color in items:
        sample_box = [15, y_offset + 2, 35, y_offset + 10]
        draw.rectangle(sample_box, outline=color, width=2)
        draw.text((42, y_offset - 2), label, fill=color, font=font)
        y_offset += 20


def visualize_and_save(img_path: str, fused_elements: list, out_vis_path: str):
    try:
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        draw_legend(draw, font)
        elem_dict = {el['id']: el for el in fused_elements}
        elements_sorted = sorted(fused_elements, key=lambda x: x.get('area', (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])), reverse=True)


        for el in elements_sorted:
            bbox = el['bbox']
            src = el['source']
            color = COLOR_SOURCE.get(src, (255, 255, 255))
            draw.rectangle(bbox, outline=color, width=1 if src in ('ocr', 'cv') else 2)

        for el in fused_elements:
            parent_info = el.get('primary_parent')
            if parent_info:
                pid = parent_info['id']
                if pid in elem_dict:
                    cb = el['bbox']
                    pb = elem_dict[pid]['bbox']
                    cc = ((cb[0] + cb[2]) / 2.0, (cb[1] + cb[3]) / 2.0)
                    pc = ((pb[0] + pb[2]) / 2.0, (pb[1] + pb[3]) / 2.0)
                    draw.line([cc, pc], fill=(220, 220, 220), width=1)

        os.makedirs(os.path.dirname(out_vis_path), exist_ok=True)
        img.save(out_vis_path)
    except Exception as e:
        logger.error(f"Error visualizing {img_path}: {e}")


def process_image(img_path: str, img_id: str, yolo_model, device: str, api_key: str, out_json_path: str, out_vis_path: str):
    if os.path.exists(out_json_path) and os.path.exists(out_vis_path):
        return True

    # 1. OCR
    with open(img_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    payload = {
        'requests': [{
            'image': {'content': img_b64},
            'features': [{'type': 'DOCUMENT_TEXT_DETECTION'}]
        }]
    }

    ocr_paras = []
    try:
        resp = requests.post(f"{API_ENDPOINT}?key={api_key}", json=payload, timeout=30)
        if resp.status_code == 200:
            res_obj = resp.json().get('responses', [{}])[0]
            ocr_paras = extract_paragraphs_from_response(res_obj, conf_threshold=0.80)
    except Exception as e:
        logger.error(f"OCR Error for {img_id}: {e}")

    ocr_nodes = []
    for idx, p in enumerate(ocr_paras):
        ocr_nodes.append({
            "id": f"ocr_{idx}",
            "source": "ocr",
            "bbox": p['bbox'],
            "confidence": p['confidence'],
            "area": float((p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1])),
            "label": p['text'],
            "text": p['text']
        })

    # 2. YOLO
    yolo_dets, _ = predict_and_filter_yolo(yolo_model, img_path, device=device, size_thresholds=SIZE_THRESHOLDS)
    small_nodes, med_nodes, large_nodes = [], [], []

    for idx, d in enumerate(yolo_dets):
        cat = d.get("size_category")
        bbox = [float(c) for c in d["bbox"]]
        area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        cls_name = str(d.get("class", "ui_element"))
        node_obj = {
            "id": f"{cat}_{idx}",
            "source": cat,
            "bbox": bbox,
            "confidence": float(d["confidence"]),
            "area": area,
            "label": cls_name,
            "text": None
        }
        if cat == "small":
            small_nodes.append(node_obj)
        elif cat == "medium":
            med_nodes.append(node_obj)
        elif cat == "large":
            large_nodes.append(node_obj)

    # 3. CV
    cv_res = detect_ui_elements(img_path, min_rect_conf=0.90, min_divider_conf=0.85)
    cv_nodes = []
    for idx, d in enumerate(cv_res['detections']):
        bx, by, bw, bh = d['bbox']
        bbox = [float(bx), float(by), float(bx + bw), float(by + bh)]
        cv_nodes.append({
            "id": f"cv_{idx}",
            "source": "cv",
            "bbox": bbox,
            "confidence": float(d.get("confidence", 1.0)),
            "area": float(bw * bh),
            "label": str(d.get("label", "rectangle_box")),
            "text": None
        })

    # 4. Pairwise relations
    all_nodes_dict = {}
    for n in ocr_nodes + small_nodes + med_nodes + large_nodes + cv_nodes:
        all_nodes_dict[n['id']] = n

    all_edges = []
    all_edges.extend(classify_small_x_cv(small_nodes, cv_nodes))
    sm_med, sm_lg = classify_small_x_yolo(small_nodes, med_nodes, large_nodes)
    all_edges.extend(sm_med)
    all_edges.extend(sm_lg)
    all_edges.extend(classify_medium_x_ocr(med_nodes, ocr_nodes))
    all_edges.extend(classify_medium_x_cv(med_nodes, cv_nodes))
    all_edges.extend(classify_medium_x_large(med_nodes, large_nodes))
    all_edges.extend(classify_cv_x_large(cv_nodes, large_nodes))
    all_edges.extend(classify_ocr_x_large(ocr_nodes, large_nodes))
    all_edges.extend(classify_large_x_large(large_nodes))

    small_boxes_np = np.array([s['bbox'] for s in small_nodes], dtype=np.float64) if small_nodes else np.empty((0, 4), dtype=np.float64)
    ocr_boxes_np = np.array([o['bbox'] for o in ocr_nodes], dtype=np.float64) if ocr_nodes else np.empty((0, 4), dtype=np.float64)
    mat_small_ocr = compute_matrix_small_ocr(small_boxes_np, ocr_boxes_np)
    small_ocr_matches = classify_small_boxes(small_nodes, ocr_nodes, mat_small_ocr)

    for m in small_ocr_matches:
        if m.get('case') in ('duplicate_ocr', 'nested_in_ocr', 'partial_overlap_review') and m.get('primary_ocr_match'):
            all_edges.append({
                "a_id": m['small_id'],
                "a_source": "small",
                "b_id": "ocr_matched",
                "b_source": "ocr",
                "case": "duplicate" if m['case'] == 'duplicate_ocr' else "a_nested_in_b",
                "metrics": m.get('metrics')
            })

    # 5. Resolve hierarchy & WBF Deduplication
    raw_hierarchy = resolve_hierarchy(all_nodes_dict, all_edges)
    fused_elements = apply_wbf_deduplication(raw_hierarchy, iou_thresh=0.85)

    # 6. Save Output JSON
    output_data = {
        "image": f"{img_id}.png",
        "total_elements": len(fused_elements),
        "total_raw_nodes": len(all_nodes_dict),
        "total_raw_edges": len(all_edges),
        "elements": fused_elements
    }

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    # 7. Save Visualization
    visualize_and_save(img_path, fused_elements, out_vis_path)
    return True


def run_pipeline_for_folder(folder_name: str, yolo_model, device: str, api_key: str):
    img_dir = os.path.join(DATASET1_DIR, folder_name)
    if not os.path.exists(img_dir):
        logger.error(f"Folder not found: {img_dir}")
        return

    json_out_dir = os.path.join(OUTPUT_ROOT_DIR, folder_name, "json")
    vis_out_dir = os.path.join(OUTPUT_ROOT_DIR, folder_name, "visualized")
    os.makedirs(json_out_dir, exist_ok=True)
    os.makedirs(vis_out_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.png')])
    logger.info(f"=== BẮT ĐẦU CHẠY COMBINE PIPELINE CHO FOLDER '{folder_name}' ({len(img_files)} ÁNH) ===")

    for idx, img_file in enumerate(img_files, 1):
        clean_id = img_file.replace('.png', '')
        img_path = os.path.join(img_dir, img_file)

        out_json = os.path.join(json_out_dir, f"{clean_id}_full_hierarchy.json")
        out_vis = os.path.join(vis_out_dir, f"{clean_id}_full_hierarchy.png")

        process_image(img_path, clean_id, yolo_model, device, api_key, out_json, out_vis)

        if idx % 50 == 0 or idx == len(img_files):
            logger.info(f" Tiến độ [{folder_name}]: {idx}/{len(img_files)} ảnh")


def main():
    logger.info("=== KÍCH HOẠT CHẠY FULL PIPELINE VÀ COMBINE CHO DATASET1 (BASELINE & INJECT) ===")
    device = detect_device()
    model_path = os.environ.get('RPA_YOLO_MODEL') or os.path.join(BASE_DIR, 'yolo', 'models', 'yl26best.pt')
    yolo_model = YOLO(model_path)
    api_key = load_api_key(KEY_FILE)

    t_start = time.time()

    # Process baseline folder
    run_pipeline_for_folder("baseline", yolo_model, device, api_key)

    # Process inject folder
    run_pipeline_for_folder("inject", yolo_model, device, api_key)

    t_total = time.time() - t_start
    logger.info(f"=== TỔNG THỜI GIAN HOÀN THÀNH PIPELINE CHO DATASET1: {t_total:.2f} GIÂY (~{t_total/60:.2f} PHÚT) ===")
    logger.info(f"Tất cả file output JSON & Visualized được lưu tại: {OUTPUT_ROOT_DIR}/")


if __name__ == "__main__":
    main()
