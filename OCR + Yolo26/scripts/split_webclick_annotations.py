#!/usr/bin/env python3
"""
Script to split data/webclick/annotations.json into:
1. data/webclick/instructions.json (Input for VLM/LLM without bbox)
2. data/webclick/ground_truth.json (Ground truth bbox for evaluation only)
"""

import sys
import json
from pathlib import Path

# Use pathlib for path resolution relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBCLICK_DIR = PROJECT_ROOT / "data" / "webclick"
ANNOTATIONS_PATH = WEBCLICK_DIR / "annotations.json"
INSTRUCTIONS_PATH = WEBCLICK_DIR / "instructions.json"
GROUND_TRUTH_PATH = WEBCLICK_DIR / "ground_truth.json"


def split_annotations():
    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(f"Original annotations file not found: {ANNOTATIONS_PATH}")

    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {ANNOTATIONS_PATH}, got {type(data).__name__}")

    instructions_list = []
    ground_truth_list = []
    seen_ids = set()

    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            raise ValueError(f"Record at index {idx} is not a dict: {sample}")

        # Verify required fields
        required_fields = ["id", "image", "instruction", "bbox", "bucket"]
        for field in required_fields:
            if field not in sample:
                raise KeyError(f"Record at index {idx} (id={sample.get('id')}) is missing required field '{field}'")

        sample_id = sample["id"]

        # Check for duplicate ID
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate id '{sample_id}' detected at index {idx}")
        seen_ids.add(sample_id)

        # Validate & clamp bbox values to [0, 1]
        raw_bbox = sample["bbox"]
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise ValueError(f"Record id={sample_id} has invalid bbox format (expected list of 4 floats): {raw_bbox}")

        clamped_bbox = []
        for c_idx, val in enumerate(raw_bbox):
            if not isinstance(val, (int, float)):
                raise TypeError(f"Record id={sample_id} bbox element {c_idx} is not a number: {val}")
            
            c_val = max(0.0, min(1.0, float(val)))
            if c_val != float(val):
                print(f"Warning: Record id={sample_id} bbox element {c_idx} ({val}) out of range [0, 1], clamped to {c_val:.4f}")
            clamped_bbox.append(round(c_val, 6))


        # Record for instructions.json (NO bbox field!)
        instructions_list.append({
            "id": sample_id,
            "image": sample["image"],
            "instruction": sample["instruction"],
            "bucket": sample["bucket"]
        })

        # Record for ground_truth.json (minimal GT: id, image, bbox)
        ground_truth_list.append({
            "id": sample_id,
            "image": sample["image"],
            "bbox": clamped_bbox
        })


    # Consistency checks
    if len(instructions_list) != len(ground_truth_list):
        raise ValueError(f"Mismatch in count: instructions ({len(instructions_list)}) vs ground truth ({len(ground_truth_list)})")

    for inst, gt in zip(instructions_list, ground_truth_list):
        if inst["id"] != gt["id"]:
            raise ValueError(f"ID mismatch between instructions and ground truth: {inst['id']} != {gt['id']}")

    # Write instructions.json
    with open(INSTRUCTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(instructions_list, f, ensure_ascii=False, indent=2)

    # Write ground_truth.json
    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(ground_truth_list, f, ensure_ascii=False, indent=2)

    total_samples = len(data)

    print("WebClick split completed.\n")
    print(f"Total samples: {total_samples}")
    print(f"Instructions: {INSTRUCTIONS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Ground truth: {GROUND_TRUTH_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Original annotation preserved: {ANNOTATIONS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    split_annotations()
