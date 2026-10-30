"""
Gemini Input Builder for WebClick Dataset.

Constructs standardized Gemini inputs and manifest without reading ground truth data.
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_instructions_map(instructions_path: Path) -> dict:
    """
    Loads task instructions from data/webclick/instructions.json.
    Returns dict mapping sample_id (int) -> {"instruction": str, "bucket": str, "image": str}.
    NOTE: Strictly loads instructions.json only. Ground truth is NEVER accessed.
    """
    if not instructions_path.exists():
        raise FileNotFoundError(f"Instructions file not found: {instructions_path}")

    with open(instructions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instructions_map = {}
    for record in data:
        sid = record.get("id")
        if sid is not None:
            instructions_map[int(sid)] = {
                "id": int(sid),
                "instruction": record.get("instruction", ""),
                "bucket": record.get("bucket", "general"),
                "image": record.get("image", "")
            }
    return instructions_map


def validate_sample_files(
    sample_id: int,
    image_dir: Path,
    perception_dir: Path,
    instructions_map: dict
) -> tuple[bool, str, dict]:
    """
    Validates presence and schema consistency of all inputs for a sample.
    Returns: (is_valid, error_msg, sample_dict)
    """
    stem = f"{sample_id:04d}"
    img_path = image_dir / f"{stem}.png"
    sample_perc_dir = perception_dir / stem
    compact_path = sample_perc_dir / f"{stem}_ui_compact.json"
    annotated_path = sample_perc_dir / f"{stem}_annotated.png"

    # 1. File existence checks
    if not img_path.exists():
        return False, f"Original image missing: {img_path}", {}

    if not compact_path.exists():
        return False, f"Compact JSON missing: {compact_path}", {}

    if not annotated_path.exists():
        return False, f"Annotated image missing: {annotated_path}", {}

    if sample_id not in instructions_map:
        return False, f"Instruction missing for sample_id {sample_id} in instructions.json", {}

    # 2. Compact JSON parsing & schema checks
    try:
        with open(compact_path, "r", encoding="utf-8") as f:
            compact_data = json.load(f)
    except Exception as e:
        return False, f"Corrupt compact JSON: {e}", {}

    if not isinstance(compact_data, dict):
        return False, "Compact JSON root is not a dict", {}

    required_keys = ["image", "width", "height", "elements"]
    for k in required_keys:
        if k not in compact_data:
            return False, f"Compact JSON missing key: {k}", {}

    # Check filename consistency
    compact_img_name = compact_data.get("image", "")
    expected_img_name = f"{stem}.png"
    if compact_img_name != expected_img_name:
        return False, f"Image filename mismatch: compact={compact_img_name} vs expected={expected_img_name}", {}

    # Check element IDs
    elements = compact_data.get("elements", [])
    if not isinstance(elements, list):
        return False, "'elements' is not a list", {}

    ids = [e.get("element_id") for e in elements if isinstance(e, dict) and "element_id" in e]
    if len(ids) != len(elements):
        return False, "Some elements are missing element_id", {}

    if not all(isinstance(i, int) for i in ids):
        return False, "Some element_ids are not integers", {}

    if len(ids) != len(set(ids)):
        return False, "Duplicate element_ids found in compact JSON", {}

    if sorted(ids) != list(range(len(elements))):
        return False, f"element_ids not continuous 0..N-1: {sorted(ids)[:5]}...", {}

    # 3. Build relative/normalized path object
    instr_info = instructions_map[sample_id]
    sample_obj = {
        "sample_id": sample_id,
        "image_path": str(img_path.relative_to(PROJECT_ROOT)),
        "annotated_image_path": str(annotated_path.relative_to(PROJECT_ROOT)),
        "compact_json_path": str(compact_path.relative_to(PROJECT_ROOT)),
        "instruction": instr_info["instruction"],
        "bucket": instr_info["bucket"]
    }

    return True, "", sample_obj


def build_sample_input(
    sample_id: int,
    image_dir: Path,
    perception_dir: Path,
    instructions_map: dict
) -> dict:
    """
    Builds a validated sample input dict for Gemini.
    Raises ValueError if validation fails.
    """
    is_valid, err_msg, sample_obj = validate_sample_files(sample_id, image_dir, perception_dir, instructions_map)
    if not is_valid:
        raise ValueError(f"Sample {sample_id} validation failed: {err_msg}")
    return sample_obj


def build_manifest(
    image_dir: Path,
    perception_dir: Path,
    instructions_path: Path,
    output_manifest_path: Path,
    sample_limit: int = None
) -> tuple[int, int, list]:
    """
    Generates manifest.jsonl containing validated Gemini input records.
    Returns: (total_samples, valid_count, invalid_samples)
    """
    instructions_map = load_instructions_map(instructions_path)
    sample_ids = sorted(instructions_map.keys())

    if sample_limit is not None and sample_limit > 0:
        sample_ids = sample_ids[:sample_limit]

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = len(sample_ids)
    valid_count = 0
    invalid_samples = []

    records = []
    for sid in sample_ids:
        is_valid, err_msg, sample_obj = validate_sample_files(sid, image_dir, perception_dir, instructions_map)
        if is_valid:
            records.append(sample_obj)
            valid_count += 1
        else:
            invalid_samples.append({"sample_id": sid, "error": err_msg})

    with open(output_manifest_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return total_samples, valid_count, invalid_samples
