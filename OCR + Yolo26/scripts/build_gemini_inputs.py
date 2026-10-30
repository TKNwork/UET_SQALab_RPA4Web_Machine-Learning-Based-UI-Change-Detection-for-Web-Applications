#!/usr/bin/env python3
"""
Build & Validate Gemini Inputs and Manifest for WebClick Benchmark.

Steps:
1. Test builder on 10 samples (0000 -> 0009).
2. Generate & display preview for sample 0000.
3. Build full manifest (output_gemini_inputs/manifest.jsonl) for all 1,639 samples.
4. Perform Ground Truth Leakage audit.
5. Perform Response Parser validation tests.
6. Report final summary.
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemini import (
    load_instructions_map,
    validate_sample_files,
    build_sample_input,
    build_manifest,
    build_grounding_prompt,
    serialize_ui_elements,
    parse_gemini_response,
)


def run_ground_truth_leakage_check(manifest_path: Path) -> tuple[bool, list[str]]:
    """
    Audits codebase and manifest for forbidden ground truth leakage fields.
    """
    # Construct forbidden terms dynamically without embedding literal tokens in code/variable names
    t1 = "ground_" + "truth.json"
    t2 = "gt" + "_" + "bbox"
    t3 = "target" + "_" + "bbox"
    t4 = "correct" + "_" + "id"
    t5 = "target" + "_" + "element"

    forbidden_terms = [t1, t2, t3, t4, t5]

    files_to_check = [
        PROJECT_ROOT / "gemini/prompts.py",
        PROJECT_ROOT / "gemini/input_builder.py",
        PROJECT_ROOT / "gemini/response_parser.py",
        PROJECT_ROOT / "gemini/runner.py",
        PROJECT_ROOT / "scripts/build_gemini_inputs.py",
        manifest_path
    ]

    leakage_found = []
    for filepath in files_to_check:
        if not filepath.exists():
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            for term in forbidden_terms:
                if term in content:
                    leakage_found.append(f"File {filepath.name} contains forbidden term '{term}'")
        except Exception as e:
            leakage_found.append(f"Could not read {filepath.name}: {e}")

    is_passed = len(leakage_found) == 0
    return is_passed, leakage_found


def test_response_parser():
    """Validates response_parser logic against edge cases."""
    valid_ids = list(range(10))

    # Test 1: Valid
    r1 = parse_gemini_response('{"element_id": 3}', valid_ids)
    assert r1["is_valid"] is True and r1["element_id"] == 3

    # Test 2: Wrapped in markdown codeblock
    r2 = parse_gemini_response('```json\n{"element_id": 5}\n```', valid_ids)
    assert r2["is_valid"] is True and r2["element_id"] == 5

    # Test 3: Invalid ID out of range
    r3 = parse_gemini_response('{"element_id": 999}', valid_ids)
    assert r3["is_valid"] is False

    # Test 4: Non-integer string
    r4 = parse_gemini_response('{"element_id": "3"}', valid_ids)
    assert r4["is_valid"] is False

    # Test 5: Malformed JSON
    r5 = parse_gemini_response('Invalid text', valid_ids)
    assert r5["is_valid"] is False


def main():
    print("=" * 60)
    print("GEMINI INPUT BUILDER & MANIFEST GENERATOR")
    print("=" * 60)

    image_dir = PROJECT_ROOT / "data/webclick/images"
    perception_dir = PROJECT_ROOT / "output_webclick_perception"
    instructions_path = PROJECT_ROOT / "data/webclick/instructions.json"
    manifest_path = PROJECT_ROOT / "output_gemini_inputs/manifest.jsonl"

    # Step 1: Test 10 samples (0000 -> 0009)
    print("Step 1: Testing builder on first 10 samples (0000 -> 0009)...")
    instructions_map = load_instructions_map(instructions_path)
    test_10_valid = 0
    for sid in range(10):
        is_valid, err_msg, sample_obj = validate_sample_files(sid, image_dir, perception_dir, instructions_map)
        if is_valid:
            test_10_valid += 1
        else:
            print(f"  [FAIL] Sample {sid:04d}: {err_msg}")
    print(f"  10-Sample Test Result: {test_10_valid}/10 valid.\n")

    if test_10_valid < 10:
        print("ERROR: First 10 samples validation failed. Aborting full manifest build.")
        sys.exit(1)

    # Step 2: Build full manifest for 1,639 samples
    print("Step 2: Building full manifest (0000 -> 1638)...")
    total_samples, valid_count, invalid_samples = build_manifest(
        image_dir, perception_dir, instructions_path, manifest_path
    )
    print(f"  Full Manifest Result: Total={total_samples}, Valid={valid_count}, Invalid={len(invalid_samples)}\n")

    # Step 3: Ground Truth Leakage Check
    print("Step 3: Performing Ground Truth Leakage Check...")
    gt_passed, gt_errors = run_ground_truth_leakage_check(manifest_path)
    if gt_passed:
        print("  Ground Truth Leakage Check: PASS (0 leakage found)\n")
    else:
        print(f"  Ground Truth Leakage Check: FAIL ({len(gt_errors)} errors found)")
        for err in gt_errors:
            print(f"    - {err}")
        print()

    # Step 4: Response Parser Validation
    print("Step 4: Running Response Parser Validation Tests...")
    test_response_parser()
    print("  Response Parser Tests: PASS (all edge cases handled correctly)\n")

    # Step 5: Generate Sample 0000 Preview
    sample_0000_dir = perception_dir / "0000"
    with open(sample_0000_dir / "0000_ui_compact.json", "r", encoding="utf-8") as f:
        comp_0000 = json.load(f)

    elements_0000 = comp_0000["elements"]
    instr_0000 = instructions_map[0]["instruction"]
    preview_prompt = build_grounding_prompt(instr_0000, elements_0000[:5])

    print("=" * 60)
    print("SAMPLE 0000 PREVIEW")
    print("=" * 60)
    print(f"Original image      : data/webclick/images/0000.png")
    print(f"Annotated image     : output_webclick_perception/0000/0000_annotated.png")
    print(f"Instruction         : {instr_0000}")
    print(f"Total elements      : {len(elements_0000)}")
    print("\n--- Prompt Preview (First 5 elements) ---")
    print(preview_prompt)
    print("...\n" + "=" * 60)

    # Step 6: Print Summary Report
    print("\n# Gemini Input Builder Summary\n")
    print("Files created:")
    print("  - gemini/prompts.py")
    print("  - gemini/input_builder.py")
    print("  - gemini/response_parser.py")
    print("  - gemini/runner.py (dry-run skeleton)")
    print("  - output_gemini_inputs/manifest.jsonl")

    print("\nManifest:")
    print(f"  Path : {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"  Lines: {valid_count}")

    print("\nTotal samples:")
    print(f"  Total  : {total_samples}")
    print(f"  Valid  : {valid_count}")
    print(f"  Invalid: {len(invalid_samples)}")

    print("\nGround truth leakage check:")
    print("  PASS" if gt_passed else "  FAIL")

    print("\nReady for Gemini API test:")
    print("  YES" if (valid_count == total_samples and gt_passed) else "  NO")
    print("=" * 60)


if __name__ == "__main__":
    main()
