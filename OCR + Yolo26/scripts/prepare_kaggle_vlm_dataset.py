#!/usr/bin/env python3
"""
Phase 1 - Prepare WebClick Dataset Package for Kaggle VLM Open-Weight Models.

Gathers, validates, and packages local WebClick assets into webclick_vlm_input/
without moving original files or introducing Ground Truth leakage.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def format_bytes(size_bytes: int) -> str:
    """Formats bytes into human-readable MB / GB string."""
    mb = size_bytes / (1024 * 1024)
    if mb >= 1024:
        gb = mb / 1024
        return f"{gb:.2f} GB ({size_bytes:,} bytes)"
    return f"{mb:.2f} MB ({size_bytes:,} bytes)"


def validate_image(img_path: Path) -> tuple[bool, int, int, str | None]:
    """Validates Pillow image file and returns (valid, width, height, error_msg)."""
    if not img_path.exists():
        return False, 0, 0, f"Image file does not exist: {img_path}"
    try:
        with Image.open(img_path) as img:
            img.verify()
        with Image.open(img_path) as img:
            w, h = img.size
        return True, w, h, None
    except Exception as e:
        return False, 0, 0, f"Corrupt image ({img_path.name}): {e}"


def validate_compact_json(json_path: Path, expected_sid: int, expected_w: int, expected_h: int) -> tuple[bool, dict | None, str | None]:
    """Validates compact UI JSON schema, element IDs, bboxes, and image dimensions."""
    if not json_path.exists():
        return False, None, f"Compact JSON missing: {json_path}"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, None, f"JSON parse error ({json_path.name}): {e}"

    if not isinstance(data, dict):
        return False, None, f"JSON root is not dict in {json_path.name}"

    for req_key in ["image", "width", "height", "elements"]:
        if req_key not in data:
            return False, None, f"Missing key '{req_key}' in {json_path.name}"

    cw = data["width"]
    ch = data["height"]

    # Image size match check
    if cw != expected_w or ch != expected_h:
        return False, None, f"Dimension mismatch in {json_path.name}: compact ({cw}x{ch}) vs image ({expected_w}x{expected_h})"

    # Image filename check
    img_field = str(data["image"])
    img_stem = Path(img_field).stem
    expected_stem = f"{expected_sid:04d}"
    if img_stem not in (str(expected_sid), expected_stem):
        return False, None, f"Image filename mismatch in {json_path.name}: field '{img_field}' vs sample {expected_sid}"

    elements = data.get("elements", [])
    if not isinstance(elements, list):
        return False, None, f"'elements' is not list in {json_path.name}"

    element_ids = []
    for idx, elem in enumerate(elements):
        if not isinstance(elem, dict):
            return False, None, f"Element #{idx} is not dict in {json_path.name}"
        if "element_id" not in elem or "bbox" not in elem:
            return False, None, f"Element #{idx} missing element_id or bbox in {json_path.name}"

        eid = elem["element_id"]
        if isinstance(eid, bool) or not isinstance(eid, int):
            return False, None, f"Element #{idx} element_id is not integer in {json_path.name}"
        element_ids.append(eid)

        bbox = elem["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False, None, f"Element #{idx} bbox is not 4-element list in {json_path.name}"

        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= cw and 0 <= y1 < y2 <= ch):
            return False, None, f"Invalid bbox [{x1}, {y1}, {x2}, {y2}] for element_id {eid} in {json_path.name} (bounds {cw}x{ch})"

    # Check ID uniqueness & contiguous range
    if len(element_ids) != len(set(element_ids)):
        return False, None, f"Duplicate element_ids found in {json_path.name}"
    if sorted(element_ids) != list(range(len(elements))):
        return False, None, f"element_ids non-contiguous or missing in {json_path.name}: {sorted(element_ids)}"

    return True, data, None


def main():
    print("=" * 70)
    print("PHASE 1: PREPARE KAGGLE VLM INPUT DATASET PACKAGE")
    print("=" * 70)

    # 1. Paths Setup
    data_dir = PROJECT_ROOT / "data" / "webclick"
    perception_dir = PROJECT_ROOT / "output_webclick_perception"
    instructions_file = data_dir / "instructions.json"

    output_dir = PROJECT_ROOT / "webclick_vlm_input"
    out_images_dir = output_dir / "images"
    out_annotated_dir = output_dir / "annotated"
    out_compact_dir = output_dir / "compact"

    for d in [out_images_dir, out_annotated_dir, out_compact_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"

    # 2. Load Instructions Map
    if not instructions_file.exists():
        print(f"[CRITICAL ERROR] Instructions file missing: {instructions_file}")
        sys.exit(1)

    with open(instructions_file, "r", encoding="utf-8") as f:
        instructions_raw = json.load(f)

    instructions_map = {}
    for item in instructions_raw:
        sid = item.get("id")
        if sid is not None:
            instructions_map[sid] = item

    print(f"Source Instructions Loaded : {len(instructions_map)} records")

    # 3. Main Processing Loop (Sample 0000 -> 1638)
    total_samples = 1639
    processed_count = 0
    skipped_count = 0

    missing_errs = 0
    corrupt_errs = 0
    id_mismatch_errs = 0
    bbox_invalid_errs = 0
    size_mismatch_errs = 0
    instruction_missing_errs = 0

    manifest_records = []

    for sid in range(total_samples):
        stem = f"{sid:04d}"
        sid_log = stem

        # Source paths
        src_orig_img = data_dir / "images" / f"{stem}.png"
        src_ann_img = perception_dir / stem / f"{stem}_annotated.png"
        src_compact_json = perception_dir / stem / f"{stem}_ui_compact.json"

        # Destination paths
        dst_orig_img = out_images_dir / f"{stem}.png"
        dst_ann_img = out_annotated_dir / f"{stem}_annotated.png"
        dst_compact_json = out_compact_dir / f"{stem}_ui_compact.json"

        # Check instruction
        instr_item = instructions_map.get(sid)
        if not instr_item or not instr_item.get("instruction") or not instr_item.get("bucket"):
            print(f"[FAIL] {sid_log} -> Instruction or bucket missing in instructions.json")
            instruction_missing_errs += 1
            continue

        instruction_text = instr_item["instruction"].strip()
        bucket_text = instr_item["bucket"].strip()

        # Validate Original Image
        orig_valid, orig_w, orig_h, orig_err = validate_image(src_orig_img)
        if not orig_valid:
            print(f"[FAIL] {sid_log} -> Original image error: {orig_err}")
            if "does not exist" in str(orig_err):
                missing_errs += 1
            else:
                corrupt_errs += 1
            continue

        # Validate Annotated Image
        ann_valid, _, _, ann_err = validate_image(src_ann_img)
        if not ann_valid:
            print(f"[FAIL] {sid_log} -> Annotated image error: {ann_err}")
            if "does not exist" in str(ann_err):
                missing_errs += 1
            else:
                corrupt_errs += 1
            continue

        # Validate Compact JSON
        compact_valid, compact_data, compact_err = validate_compact_json(src_compact_json, sid, orig_w, orig_h)
        if not compact_valid:
            print(f"[FAIL] {sid_log} -> Compact JSON error: {compact_err}")
            if "missing" in str(compact_err) or "does not exist" in str(compact_err):
                missing_errs += 1
            elif "Dimension mismatch" in str(compact_err):
                size_mismatch_errs += 1
            elif "mismatch" in str(compact_err):
                id_mismatch_errs += 1
            elif "bbox" in str(compact_err):
                bbox_invalid_errs += 1
            else:
                corrupt_errs += 1
            continue

        # Check if destination files already exist & valid (Idempotency check)
        is_already_copied = (
            dst_orig_img.exists() and dst_orig_img.stat().st_size == src_orig_img.stat().st_size and
            dst_ann_img.exists() and dst_ann_img.stat().st_size == src_ann_img.stat().st_size and
            dst_compact_json.exists() and dst_compact_json.stat().st_size == src_compact_json.stat().st_size
        )

        if is_already_copied:
            skipped_count += 1
            status_str = "SKIP"
        else:
            # Copy files using shutil.copy2
            shutil.copy2(src_orig_img, dst_orig_img)
            shutil.copy2(src_ann_img, dst_ann_img)
            shutil.copy2(src_compact_json, dst_compact_json)
            processed_count += 1
            status_str = "DONE"

        # Manifest Record (relative paths only)
        record = {
            "sample_id": sid,
            "image": f"images/{stem}.png",
            "annotated": f"annotated/{stem}_annotated.png",
            "compact": f"compact/{stem}_ui_compact.json",
            "instruction": instruction_text,
            "bucket": bucket_text
        }
        manifest_records.append(record)

        if sid % 100 == 0 or sid == total_samples - 1:
            print(f"[{status_str} ] {sid_log} | original={orig_w}x{orig_h} | elements={len(compact_data['elements'])}")

    # 4. Write manifest.jsonl
    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in manifest_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 5. Ground Truth Leakage Audit
    print("\n" + "=" * 70)
    print("AUDITING FOR GROUND TRUTH LEAKAGE IN webclick_vlm_input/")
    print("=" * 70)

    forbidden_filenames = ["ground_truth.json", "annotations.json", "gt_bbox", "target_bbox"]
    forbidden_json_keys = ["ground_truth", "annotations", "gt_bbox", "target_bbox", "correct_id"]

    gt_leakage_detected = False
    leakage_details = []

    for root, dirs, files in os.walk(output_dir):
        for fname in files:
            fpath = Path(root) / fname
            # Check filename
            if any(forbidden in fname.lower() for forbidden in forbidden_filenames):
                gt_leakage_detected = True
                leakage_details.append(f"Forbidden file found: {fpath}")

            # Check JSON contents for forbidden keys
            if fname.endswith(".json") or fname.endswith(".jsonl"):
                try:
                    content_str = fpath.read_text(encoding="utf-8").lower()
                    for fkey in forbidden_json_keys:
                        if f'"{fkey}"' in content_str:
                            gt_leakage_detected = True
                            leakage_details.append(f"Forbidden key '{fkey}' in file: {fpath}")
                except Exception as e:
                    leakage_details.append(f"Could not read file {fpath}: {e}")

    gt_leakage_status = "FAIL" if gt_leakage_detected else "PASS"
    if gt_leakage_detected:
        print("[WARNING] Ground Truth Leakage Detected!")
        for detail in leakage_details:
            print(f"  - {detail}")
    else:
        print("[PASS] Zero Ground Truth leakage detected in webclick_vlm_input/")

    # 6. Final Validation & Size Calculation
    img_files = list(out_images_dir.glob("*.png"))
    ann_files = list(out_annotated_dir.glob("*.png"))
    compact_files = list(out_compact_dir.glob("*.json"))

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_line_count = sum(1 for line in f if line.strip())

    total_failures = (
        missing_errs + corrupt_errs + id_mismatch_errs +
        bbox_invalid_errs + size_mismatch_errs + instruction_missing_errs
    )

    # Sizes
    images_bytes = sum(f.stat().st_size for f in img_files)
    ann_bytes = sum(f.stat().st_size for f in ann_files)
    compact_bytes = sum(f.stat().st_size for f in compact_files)
    manifest_bytes = manifest_path.stat().st_size if manifest_path.exists() else 0
    total_package_bytes = images_bytes + ann_bytes + compact_bytes + manifest_bytes

    ready_for_phase_2 = (
        len(img_files) == total_samples and
        len(ann_files) == total_samples and
        len(compact_files) == total_samples and
        manifest_line_count == total_samples and
        total_failures == 0 and
        gt_leakage_status == "PASS"
    )

    # 7. Print Final Report
    print("\n" + "=" * 70)
    print("PHASE 1 - WEBCLICK VLM INPUT PACKAGE SUMMARY REPORT")
    print("=" * 70)
    print("\n# Phase 1 - WebClick VLM Input Package\n")
    print("Output folder:")
    print("webclick_vlm_input/\n")
    print(f"Total samples:\n{total_samples}\n")
    print(f"Original images:\n{len(img_files)}\n")
    print(f"Annotated images:\n{len(ann_files)}\n")
    print(f"Compact JSON:\n{len(compact_files)}\n")
    print(f"Manifest records:\n{manifest_line_count}\n")

    print(f"Missing:\n{missing_errs}\n")
    print(f"Corrupt:\n{corrupt_errs}\n")
    print(f"ID mismatch:\n{id_mismatch_errs}\n")
    print(f"BBox invalid:\n{bbox_invalid_errs}\n")
    print(f"Image size mismatch:\n{size_mismatch_errs}\n")
    print(f"Instruction missing:\n{instruction_missing_errs}\n")

    print(f"Ground truth leakage:\n{gt_leakage_status}\n")

    print(f"Package size:")
    print(f"Total Package : {format_bytes(total_package_bytes)}")
    print(f"Original Images: {format_bytes(images_bytes)}")
    print(f"Annotated Images: {format_bytes(ann_bytes)}")
    print(f"Compact JSON   : {format_bytes(compact_bytes)}")
    print(f"Manifest JSONL : {format_bytes(manifest_bytes)}\n")

    print("# Folder structure\n")
    print("webclick_vlm_input/")
    print("├── images/")
    print("├── annotated/")
    print("├── compact/")
    print("└── manifest.jsonl\n")

    # Sample 0000 Preview
    sample_0 = manifest_records[0] if manifest_records else {}
    print("# Sample 0000\n")
    print(f"image:\n{sample_0.get('image')}\n")
    print(f"annotated:\n{sample_0.get('annotated')}\n")
    print(f"compact:\n{sample_0.get('compact')}\n")
    print(f"instruction:\n{sample_0.get('instruction')}\n")
    print(f"bucket:\n{sample_0.get('bucket')}\n")

    print("# Ready for Phase 2 Kaggle upload")
    print("YES" if ready_for_phase_2 else "NO")
    print("=" * 70)

    if not ready_for_phase_2:
        print("[ERROR] Phase 1 validation failed. Stopping before Phase 2.")
        sys.exit(1)


if __name__ == "__main__":
    main()
