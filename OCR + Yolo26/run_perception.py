#!/usr/bin/env python3
"""
Perception Pipeline Entrypoint for UI Detection & Hierarchy Resolution.
Screenshot -> Structured UI JSON & Compact UI JSON.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Set project root using pathlib
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocr.test_gcv_ocr import run_ocr
from yolo.test_yolo_inference import run_yolo
from cv.contour_ui_detector import run_cv
from combine._core.schema import (
    normalize_ocr_nodes,
    normalize_yolo_nodes,
    normalize_cv_nodes
)
from combine.small_x_cv.classify import classify_small_x_cv
from combine.small_x_yolo.classify import classify_small_x_yolo
from combine.medium_x_ocr.classify import classify_medium_x_ocr
from combine.medium_x_cv.classify import classify_medium_x_cv
from combine.medium_x_large.classify import classify_medium_x_large
from combine.cv_x_large.classify import classify_cv_x_large
from combine.ocr_x_large.classify import classify_ocr_x_large
from combine.large_x_large.classify import classify_large_x_large
from combine.small_x_ocr.classify import classify_small_boxes
from combine._core.matrix import compute_all_matrices

import importlib.util
spec = importlib.util.spec_from_file_location("deduplicate_module", PROJECT_ROOT / "eval_groundtruth" / "01_deduplicate.py")
dedup_module = importlib.util.module_from_spec(spec)
sys.modules["deduplicate_module"] = dedup_module
spec.loader.exec_module(dedup_module)
apply_wbf_deduplication = dedup_module.apply_wbf_deduplication

from combine.full_pipeline.hierarchy_resolver import resolve_fused_hierarchy



def draw_fused_visualization(image_path: Path, elements: list, output_path: Path):
    """Vẽ tất cả các fused UI elements lên ảnh gốc cho mục đích debug."""
    try:
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        colors = {
            "ocr": (0, 200, 0),
            "small": (255, 165, 0),
            "medium": (30, 144, 255),
            "large": (148, 0, 211),
            "cv": (255, 20, 147)
        }

        for elem in elements:
            bbox = elem.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                src = elem.get('source', 'medium')
                color = colors.get(src, (0, 255, 255))
                
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                label = f"{elem['id']} ({src})"
                if elem.get('text'):
                    label += f": {elem['text'][:15]}"
                    
                text_pos = (x1, max(0, y1 - 12))
                tb = draw.textbbox(text_pos, label, font=font) if hasattr(draw, 'textbbox') else None
                if tb:
                    draw.rectangle(tb, fill=(0, 0, 0))
                draw.text(text_pos, label, fill=(255, 255, 255), font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")


def draw_omniparser_annotation(image_path: Path, elements: list, output_path: Path):
    """
    Vẽ ảnh annotated phong cách OmniParser dành cho VLM / Gemini Pro:
    - Bounding Box viền rõ nét quanh từng phần tử.
    - Numeric Element ID Badge (thẻ số tròn/vuông nổi bật ở góc trên-trái) chỉ chứa duy nhất element_id.
    - Tuyệt đối không chứa text dài gây clutter.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        colors = [
            (235, 50, 50),    # Red
            (30, 144, 255),   # Blue
            (46, 139, 87),    # Green
            (148, 0, 211),    # Purple
            (255, 140, 0),    # Orange
            (220, 20, 60)     # Crimson
        ]

        for elem in elements:
            bbox = elem.get('bbox', [])
            elem_id = elem.get('element_id')
            if len(bbox) == 4 and elem_id is not None:
                x1, y1, x2, y2 = bbox
                color = colors[elem_id % len(colors)]
                
                # 1. Bounding box viền 2px
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # 2. Numeric Badge (chỉ chứa duy nhất số element_id)
                badge_text = str(elem_id)
                text_pos = (x1, max(0, y1 - 12))
                
                tb = draw.textbbox(text_pos, badge_text, font=font) if hasattr(draw, 'textbbox') else None
                if tb:
                    bg_rect = [tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1]
                    draw.rectangle(bg_rect, fill=color)
                    draw.text(text_pos, badge_text, fill=(255, 255, 255), font=font)
                else:
                    draw.text(text_pos, badge_text, fill=color, font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
    except Exception as e:
        print(f"Warning: Could not create OmniParser annotation: {e}")


def process_perception(image_path: Path, output_dir: Path, visualize: bool = False, save_debug: bool = False):
    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)

    img_name = image_path.name
    base_stem = image_path.stem

    # Lấy thông tin kích thước ảnh
    with Image.open(image_path) as img:
        img_w, img_h = img.size

    print("=" * 60)
    print("PERCEPTION PIPELINE")
    print("=" * 60)
    print(f"Image: {img_name}")
    print(f"Size: {img_w} x {img_h}\n")

    # [1/6] Running OCR
    print("[1/6] Running OCR...")
    raw_ocr = run_ocr(str(image_path))
    ocr_nodes = normalize_ocr_nodes(raw_ocr)
    print(f"      {len(ocr_nodes)} detections\n")

    # [2/6] Running YOLO
    print("[2/6] Running YOLO...")
    raw_small, raw_medium, raw_large = run_yolo(str(image_path))
    small_nodes = normalize_yolo_nodes(raw_small, "small")
    med_nodes = normalize_yolo_nodes(raw_medium, "medium")
    large_nodes = normalize_yolo_nodes(raw_large, "large")
    total_yolo = len(small_nodes) + len(med_nodes) + len(large_nodes)
    print(f"      Raw YOLO: {total_yolo}")
    print(f"      Small: {len(small_nodes)}")
    print(f"      Medium: {len(med_nodes)}")
    print(f"      Large: {len(large_nodes)}\n")

    # [3/6] Running OpenCV
    print("[3/6] Running OpenCV...")
    raw_cv = run_cv(str(image_path))
    cv_nodes = normalize_cv_nodes(raw_cv)
    rect_count = sum(1 for c in cv_nodes if c.get('class_label') == 'rectangle_box')
    div_count = sum(1 for c in cv_nodes if c.get('class_label') == 'divider')
    print(f"      Rectangles: {rect_count}")
    print(f"      Dividers: {div_count}\n")

    # Gom toàn bộ raw nodes
    all_raw_nodes = ocr_nodes + small_nodes + med_nodes + large_nodes + cv_nodes
    total_raw_nodes = len(all_raw_nodes)

    # [4/6] Running pairwise relationship modules
    print("[4/6] Running pairwise relationship modules...")
    raw_edges = []
    raw_edges.extend(classify_small_x_cv(small_nodes, cv_nodes))
    s_m_edges, s_l_edges = classify_small_x_yolo(small_nodes, med_nodes, large_nodes)
    raw_edges.extend(s_m_edges)
    raw_edges.extend(s_l_edges)
    raw_edges.extend(classify_medium_x_ocr(med_nodes, ocr_nodes))
    raw_edges.extend(classify_medium_x_cv(med_nodes, cv_nodes))
    raw_edges.extend(classify_medium_x_large(med_nodes, large_nodes))
    raw_edges.extend(classify_cv_x_large(cv_nodes, large_nodes))
    raw_edges.extend(classify_ocr_x_large(ocr_nodes, large_nodes))
    raw_edges.extend(classify_large_x_large(large_nodes))

    if small_nodes and ocr_nodes:
        boxes_s = [s['bbox'] for s in small_nodes]
        boxes_o = [o['bbox'] for o in ocr_nodes]
        import numpy as np
        m_dict = compute_all_matrices(np.array(boxes_s, dtype=np.float64), np.array(boxes_o, dtype=np.float64))
        s_ocr_res = classify_small_boxes(small_nodes, ocr_nodes, m_dict)
        for r in s_ocr_res:
            if r.get('case') in ('duplicate_ocr', 'nested_in_ocr', 'partial_overlap_review'):
                raw_edges.append({
                    "a_id": r['small_id'],
                    "a_source": "small",
                    "b_id": "ocr_matched",
                    "b_source": "ocr",
                    "case": "duplicate" if r['case'] == 'duplicate_ocr' else "a_nested_in_b",
                    "metrics": r.get('metrics')
                })

    print(f"      Raw nodes: {total_raw_nodes}")
    print(f"      Raw edges: {len(raw_edges)}\n")

    # [5/6] Running WBF
    print("[5/6] Running WBF...")
    fused_elements = apply_wbf_deduplication(all_raw_nodes, iou_thresh=0.85)
    print(f"      Fused elements: {len(fused_elements)}\n")

    # [6/6] Resolving hierarchy
    print("[6/6] Resolving hierarchy...")
    resolved_elements = resolve_fused_hierarchy(fused_elements)

    # Validation element_id bắt buộc
    ids = [e["element_id"] for e in resolved_elements]
    assert len(ids) == len(set(ids)), f"ERROR: Duplicate element_id detected in {img_name}!"
    assert sorted(ids) == list(range(len(resolved_elements))), f"ERROR: Element IDs are not continuous 0..{len(resolved_elements)-1} in {img_name}!"

    root_count = sum(1 for e in resolved_elements if e['primary_parent'] is None)
    parent_child_links = sum(len(e['children']) for e in resolved_elements)
    gap_count = sum(1 for e in resolved_elements if e['hierarchy_gap'])

    print(f"      Root elements: {root_count}")
    print(f"      Parent-child links: {parent_child_links}")
    print(f"      Hierarchy gaps: {gap_count}\n")

    # Đóng gói Full Structured UI JSON
    full_json_data = {
        "image": {
            "file_name": img_name,
            "width": img_w,
            "height": img_h
        },
        "summary": {
            "total_raw_nodes": total_raw_nodes,
            "total_raw_edges": len(raw_edges),
            "total_elements": len(resolved_elements)
        },
        "elements": resolved_elements
    }

    # Đóng gói Compact UI JSON dành cho VLM / Gemini Pro
    compact_elements = []
    for elem in resolved_elements:
        p_num = elem['primary_parent']['element_id'] if elem['primary_parent'] else None
        compact_elements.append({
            "element_id": elem['element_id'],
            "text": elem['text'],
            "class_label": elem['class_label'],
            "bbox": elem['bbox'],
            "parent": p_num
        })

    compact_json_data = {
        "image": img_name,
        "width": img_w,
        "height": img_h,
        "elements": compact_elements
    }

    # Lưu kết quả
    sample_out_dir = output_dir / base_stem
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    json_full_path = sample_out_dir / f"{base_stem}_ui.json"
    with open(json_full_path, "w", encoding="utf-8") as f:
        json.dump(full_json_data, f, ensure_ascii=False, indent=2)

    json_compact_path = sample_out_dir / f"{base_stem}_ui_compact.json"
    with open(json_compact_path, "w", encoding="utf-8") as f:
        json.dump(compact_json_data, f, ensure_ascii=False, indent=2)

    vis_path = None
    annotated_path = None
    if visualize:
        vis_path = sample_out_dir / f"{base_stem}_visualized.png"
        draw_fused_visualization(image_path, resolved_elements, vis_path)

        annotated_path = sample_out_dir / f"{base_stem}_annotated.png"
        draw_omniparser_annotation(image_path, resolved_elements, annotated_path)


    if save_debug:
        debug_dir = sample_out_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_dir / "raw_nodes.json", "w", encoding="utf-8") as f:
            json.dump(all_raw_nodes, f, ensure_ascii=False, indent=2)
        with open(debug_dir / "raw_edges.json", "w", encoding="utf-8") as f:
            json.dump(raw_edges, f, ensure_ascii=False, indent=2)

    print("Saved:")
    print(f"{json_full_path.relative_to(PROJECT_ROOT)}")
    print(f"{json_compact_path.relative_to(PROJECT_ROOT)}")
    if annotated_path:
        print(f"{annotated_path.relative_to(PROJECT_ROOT)}")
    if vis_path:
        print(f"{vis_path.relative_to(PROJECT_ROOT)}")




def main():
    parser = argparse.ArgumentParser(description="Run Perception Pipeline on a single image.")
    parser.add_argument("--image", required=True, type=str, help="Path to input screenshot image.")
    parser.add_argument("--output", type=str, default="output_perception", help="Output base directory.")
    parser.add_argument("--visualize", action="store_true", help="Generate visualized debug image.")
    parser.add_argument("--save-debug", action="store_true", help="Save intermediate raw nodes & edges debug files.")

    args = parser.parse_args()
    image_path = Path(args.image).resolve()
    output_dir = PROJECT_ROOT / args.output

    process_perception(image_path, output_dir, visualize=args.visualize, save_debug=args.save_debug)


if __name__ == "__main__":
    main()
