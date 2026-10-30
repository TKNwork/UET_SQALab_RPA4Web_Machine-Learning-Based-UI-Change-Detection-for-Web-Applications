#!/usr/bin/env python3
"""
Generate OmniParser-style Annotated Screenshots for WebClick Dataset.

Input:
- Original screenshots: data/webclick/images/<sample_id>.png
- Perception Compact JSON: output_webclick_perception/<sample_id>/<sample_id>_ui_compact.json

Output:
- Annotated screenshot: output_webclick_perception/<sample_id>/<sample_id>_annotated.png

Rules:
- NO rerun detection (no OCR, YOLO, CV).
- Strict ID alignment (uses element_id directly from compact JSON).
- BBox validation (0 <= x1 < x2 <= width, 0 <= y1 < y2 <= height).
- Image size metadata consistency check.
- Resume support (skips valid annotated PNGs).
- Error logging to output_webclick_perception/annotation_errors.log.
"""

import os
import sys
import json
import datetime
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def log_annotation_error(log_file: Path, stem: str, error_type: str, message: str):
    """Write error entry to output_webclick_perception/annotation_errors.log"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] | Sample: {stem} | ErrorType: {error_type} | Message: {message}\n")


def validate_sample_for_annotation(img_path: Path, compact_path: Path) -> tuple[bool, str, str, dict]:
    """
    Validates original image and compact JSON before rendering.
    Returns: (is_valid, error_type, error_msg, compact_data)
    """
    if not img_path.exists():
        return False, "image_missing", f"Original image missing: {img_path}", {}
    if not compact_path.exists():
        return False, "compact_json_missing", f"Compact JSON missing: {compact_path}", {}

    try:
        with open(compact_path, "r", encoding="utf-8") as f:
            compact_data = json.load(f)
    except Exception as e:
        return False, "corrupt_json", f"Failed to parse compact JSON: {e}", {}

    if not isinstance(compact_data, dict):
        return False, "malformed_json", "Compact JSON is not a dict", {}

    for k in ["image", "width", "height", "elements"]:
        if k not in compact_data:
            return False, "schema_error", f"Missing root key: {k}", {}

    elements = compact_data["elements"]
    if not isinstance(elements, list):
        return False, "schema_error", "'elements' is not a list", {}

    # 1. Image metadata check
    try:
        with Image.open(img_path) as img:
            actual_w, actual_h = img.size
    except Exception as e:
        return False, "corrupt_image", f"Cannot open original image: {e}", {}

    comp_w = compact_data["width"]
    comp_h = compact_data["height"]
    if actual_w != comp_w or actual_h != comp_h:
        return False, "dimension_mismatch", f"Dimension mismatch: actual=({actual_w}x{actual_h}) vs compact=({comp_w}x{comp_h})", {}

    # 2. Element and BBox validation
    ids = []
    for elem in elements:
        if not isinstance(elem, dict):
            return False, "element_error", "Element is not a dict", {}
        if "element_id" not in elem or "bbox" not in elem:
            return False, "element_error", "Element missing element_id or bbox", {}

        eid = elem["element_id"]
        if not isinstance(eid, int):
            return False, "id_type_error", f"element_id is not int: {type(eid)} ({eid})", {}
        ids.append(eid)

        bbox = elem["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False, "bbox_format_error", f"Invalid bbox format: {bbox}", {}

        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= actual_w and 0 <= y1 < y2 <= actual_h):
            return False, "bbox_out_of_bounds", f"BBox out of bounds: elem={eid}, bbox={bbox}, img=({actual_w}x{actual_h})", {}

    # 3. ID uniqueness and 0..N-1 continuity
    if len(ids) != len(set(ids)):
        return False, "duplicate_ids", f"Duplicate element_ids found in {compact_path.name}", {}

    if sorted(ids) != list(range(len(elements))):
        return False, "non_continuous_ids", f"element_ids not sorted 0..N-1 in {compact_path.name}", {}

    return True, "", "", compact_data


def is_annotated_valid(annotated_path: Path) -> bool:
    """Checks if annotated PNG exists and is a valid readable image."""
    if not annotated_path.exists() or annotated_path.stat().st_size == 0:
        return False
    try:
        with Image.open(annotated_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def render_omniparser_annotation(img_path: Path, compact_data: dict, output_path: Path):
    """
    Renders OmniParser-style annotated image:
    - Bounding box outline around each element.
    - Numeric element_id badge near top-left of bbox.
    - No clutter text.
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img_w, img_h = img.size
        draw = ImageDraw.Draw(img)

        # High-contrast color palette
        colors = [
            (230, 25, 75),    # Red
            (60, 180, 75),    # Green
            (255, 225, 25),   # Yellow
            (0, 130, 200),    # Blue
            (245, 130, 48),   # Orange
            (145, 30, 180),   # Purple
            (70, 240, 240),   # Cyan
            (240, 50, 230),   # Magenta
            (210, 245, 60),   # Lime
            (250, 190, 212),  # Pink
            (0, 128, 128),    # Teal
            (220, 190, 255),  # Lavender
        ]

        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        elements = compact_data["elements"]
        for elem in elements:
            elem_id = elem["element_id"]
            x1, y1, x2, y2 = elem["bbox"]
            color = colors[elem_id % len(colors)]

            # 1. Outline rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # 2. Numeric Badge (only element_id)
            badge_text = str(elem_id)

            if hasattr(draw, "textbbox"):
                tb = draw.textbbox((0, 0), badge_text, font=font)
                tw = tb[2] - tb[0]
                th = tb[3] - tb[1]
            else:
                tw, th = (8 * len(badge_text), 12)

            pad_x = 3
            pad_y = 2
            badge_w = tw + pad_x * 2
            badge_h = th + pad_y * 2

            badge_x = float(x1)
            if y1 - badge_h >= 0:
                badge_y = float(y1 - badge_h)
            else:
                badge_y = float(y1)

            # Clamp badge to image boundaries
            badge_x = max(0.0, min(float(img_w - badge_w), badge_x))
            badge_y = max(0.0, min(float(img_h - badge_h), badge_y))

            bg_rect = [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h]
            draw.rectangle(bg_rect, fill=color)

            # Contrast text fill
            r, g, b = color
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            text_fill = (0, 0, 0) if brightness > 165 else (255, 255, 255)

            draw.text((badge_x + pad_x, badge_y + pad_y), badge_text, fill=text_fill, font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "PNG")


def main():
    parser = argparse.ArgumentParser(description="Generate WebClick OmniParser-style Annotated Images.")
    parser.add_argument("--images-dir", type=str, default="data/webclick/images", help="Directory containing original images.")
    parser.add_argument("--output-dir", type=str, default="output_webclick_perception", help="Perception output base directory.")

    args = parser.parse_args()

    images_dir = PROJECT_ROOT / args.images_dir
    output_dir = PROJECT_ROOT / args.output_dir
    error_log_file = output_dir / "annotation_errors.log"

    valid_exts = ('.png', '.jpg', '.jpeg')
    image_files = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
    total_samples = len(image_files)

    print("=" * 60)
    print("WEBCLICK ANNOTATED SCREENSHOT GENERATOR")
    print("=" * 60)
    print(f"Total dataset images found : {total_samples}")
    print(f"Input image directory      : {images_dir.relative_to(PROJECT_ROOT)}/")
    print(f"Output perception directory : {output_dir.relative_to(PROJECT_ROOT)}/\n")

    generated_count = 0
    skipped_count = 0
    failed_count = 0
    failed_samples = []

    for idx, img_path in enumerate(image_files, 1):
        stem = img_path.stem
        sample_out_dir = output_dir / stem
        compact_path = sample_out_dir / f"{stem}_ui_compact.json"
        annotated_path = sample_out_dir / f"{stem}_annotated.png"

        if is_annotated_valid(annotated_path):
            # Cần validate xem JSON hiện tại có hợp lệ không trước khi count skip
            is_valid, err_type, err_msg, _ = validate_sample_for_annotation(img_path, compact_path)
            if is_valid:
                print(f"[SKIP] {stem} annotated already exists")
                skipped_count += 1
                continue

        # Process sample
        print(f"[RUN ] {stem}")
        is_valid, err_type, err_msg, compact_data = validate_sample_for_annotation(img_path, compact_path)

        if not is_valid:
            print(f"[FAIL] {stem} - {err_type}: {err_msg}")
            log_annotation_error(error_log_file, stem, err_type, err_msg)
            failed_count += 1
            failed_samples.append(stem)
            continue

        try:
            render_omniparser_annotation(img_path, compact_data, annotated_path)
            box_count = len(compact_data.get("elements", []))
            print(f"[DONE] {stem} - {box_count} boxes")
            generated_count += 1
        except Exception as e:
            print(f"[FAIL] {stem} - render_error: {e}")
            log_annotation_error(error_log_file, stem, "render_error", str(e))
            failed_count += 1
            failed_samples.append(stem)

    remaining_count = total_samples - (skipped_count + generated_count)

    print("\n" + "=" * 60)
    print("ANNOTATION BATCH PROGRESS SUMMARY")
    print("=" * 60)
    print(f"Total samples : {total_samples}")
    print(f"Generated     : {generated_count}")
    print(f"Skipped       : {skipped_count}")
    print(f"Failed        : {failed_count}")
    print(f"Remaining     : {remaining_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
