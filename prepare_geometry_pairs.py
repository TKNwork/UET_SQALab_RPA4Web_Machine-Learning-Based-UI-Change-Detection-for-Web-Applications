#!/usr/bin/env python3
"""
Script: prepare_geometry_pairs.py
Purpose: Collect and organize Before-After Object Detection JSON pairs 
         for Geometry Change Detection evaluation.
"""

import json
import argparse
import os
import sys
import shutil
from pathlib import Path

# Configured paths
BASE_DIR = Path(__file__).resolve().parent / 'output_dataset1'
BASELINE_DIR = BASE_DIR / "baseline" / "json"
INJECT_DIR = BASE_DIR / "inject" / "json"
OUTPUT_DIR = BASE_DIR.parent / 'outputs' / 'prepared_geometry_pairs'

# List of Geometry IDs specified for evaluation
GEOMETRY_IDS = [
    "000", "004", "012", "016", "021", "024", "028", "031", "033", "042",
    "045", "048", "050", "053", "055", "056", "061", "062", "063", "067",
    "081", "088", "092", "097", "105", "106", "112", "114", "118", "121",
    "123", "125", "126", "128", "131", "134", "141", "143", "148", "153",
    "155", "156", "160", "163", "169", "172", "176", "187", "192", "199",
    "207", "208", "213", "214", "218", "219", "221", "225", "226", "229",
    "230", "237", "239", "240", "251", "252", "253", "254", "255", "261",
    "263", "264", "269", "271", "272", "274", "275", "276", "279", "280",
    "283", "286", "291", "292", "293", "299", "302", "311", "314", "317",
    "324", "330", "332", "335", "337", "344", "345", "346", "347", "348",
    "350", "355", "363", "365", "368", "374", "379", "390", "394", "395",
    "404", "407", "411", "418", "419", "423", "426", "429", "437", "446",
    "451", "452", "456", "467", "471", "483", "485", "488", "489", "492",
    "498"
]


def validate_sources(baseline_dir: Path, inject_dir: Path) -> None:
    """Ensure source directories exist before execution."""
    if not baseline_dir.exists() or not baseline_dir.is_dir():
        print(f"[ERROR] Baseline directory does not exist: {baseline_dir}", file=sys.stderr)
        sys.exit(1)
    if not inject_dir.exists() or not inject_dir.is_dir():
        print(f"[ERROR] Inject directory does not exist: {inject_dir}", file=sys.stderr)
        sys.exit(1)


def check_id_list(raw_ids: list[str]) -> list[str]:
    """Validate ID list, format to 3 digits, and check for duplicates."""
    formatted_ids = [str(x).strip().zfill(3) for x in raw_ids]
    seen = set()
    duplicates = []
    unique_ids = []

    for item in formatted_ids:
        if item in seen:
            duplicates.append(item)
        else:
            seen.add(item)
            unique_ids.append(item)

    if duplicates:
        print(f"[WARNING] Duplicate IDs found in input list: {sorted(set(duplicates))}")

    print(f"[PRE-CHECK] Total requested IDs: {len(raw_ids)}")
    print(f"[PRE-CHECK] Actual unique Geometry IDs to process: {len(unique_ids)}")
    return unique_ids


def find_matching_files(directory: Path, id_str: str) -> list[Path]:
    """Find files starting with <ID>_ and ending with _full_hierarchy.json."""
    pattern = f"{id_str}_*_full_hierarchy.json"
    matches = [f for f in directory.glob(pattern) if f.is_file()]
    return sorted(matches)


def main():
    global BASELINE_DIR, INJECT_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path(os.environ.get('RPA_BASELINE', BASELINE_DIR)))
    parser.add_argument('--inject', type=Path, default=Path(os.environ.get('RPA_INJECT', INJECT_DIR)))
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    BASELINE_DIR, INJECT_DIR, OUTPUT_DIR = args.baseline.resolve(), args.inject.resolve(), args.output.resolve()
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        parser.error('Choose a new/empty output directory; existing research files are protected.')
    print("=== PRE-CHECK & VALIDATION ===")
    validate_sources(BASELINE_DIR, INJECT_DIR)
    target_ids = check_id_list(GEOMETRY_IDS)
    
    # Create output root directory if not exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing_baseline_ids = []
    missing_inject_ids = []
    ambiguous_ids = []
    complete_pairs = []

    print("\n=== PROCESSING PAIRS ===")
    for id_str in target_ids:
        b_files = find_matching_files(BASELINE_DIR, id_str)
        i_files = find_matching_files(INJECT_DIR, id_str)

        # Check for ambiguity
        is_ambiguous = False
        if len(b_files) > 1:
            print(f"[WARNING] Multiple baseline files for {id_str}: {[f.name for f in b_files]}")
            is_ambiguous = True
        if len(i_files) > 1:
            print(f"[WARNING] Multiple inject files for {id_str}: {[f.name for f in i_files]}")
            is_ambiguous = True

        if is_ambiguous:
            ambiguous_ids.append(id_str)
            continue

        # Check for missing files
        has_missing = False
        if len(b_files) == 0:
            print(f"[WARNING] Missing baseline: {id_str}")
            missing_baseline_ids.append(id_str)
            has_missing = True
        if len(i_files) == 0:
            print(f"[WARNING] Missing inject: {id_str}")
            missing_inject_ids.append(id_str)
            has_missing = True

        if has_missing:
            continue

        # Valid single match on both sides
        b_file = b_files[0]
        i_file = i_files[0]

        pair_dir = OUTPUT_DIR / id_str
        pair_dir.mkdir(parents=True, exist_ok=True)

        before_dest = pair_dir / "before.json"
        after_dest = pair_dir / "after.json"

        # Copy files (preserving metadata, no move/delete)
        shutil.copy2(b_file, before_dest)
        shutil.copy2(i_file, after_dest)

        complete_pairs.append({
            "id": id_str,
            "original_filename": b_file.name,
            "before": f"{id_str}/before.json",
            "after": f"{id_str}/after.json"
        })

    # Create manifest.json
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_data = {
        "change_type": "geometry",
        "total_pairs": len(complete_pairs),
        "pairs": complete_pairs
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    # Print Summary
    print("\n==============================")
    print("GEOMETRY DATASET SUMMARY")
    print("==============================")
    print(f"Requested IDs: {len(target_ids)}")
    print(f"Complete pairs: {len(complete_pairs)}")
    print(f"Missing baseline: {len(missing_baseline_ids)}")
    print(f"Missing inject: {len(missing_inject_ids)}")
    print(f"Ambiguous IDs: {len(ambiguous_ids)}")
    print("==============================")

    if missing_baseline_ids:
        print(f"Missing baseline IDs: {missing_baseline_ids}")
    if missing_inject_ids:
        print(f"Missing inject IDs: {missing_inject_ids}")
    if ambiguous_ids:
        print(f"Ambiguous IDs: {ambiguous_ids}")

    # Post-copy Verification
    print("\n=== POST-COPY VERIFICATION ===")
    pair_folders = [d for d in OUTPUT_DIR.iterdir() if d.is_dir()]
    valid_filesystem_pairs = 0

    for folder in sorted(pair_folders):
        b_json = folder / "before.json"
        a_json = folder / "after.json"
        files_in_folder = list(folder.iterdir())
        
        if len(files_in_folder) == 2 and b_json.exists() and a_json.exists():
            valid_filesystem_pairs += 1
        else:
            print(f"[ERROR] Pair folder {folder.name} structure invalid: {[f.name for f in files_in_folder]}", file=sys.stderr)

    print(f"[VERIFY] Valid pair folders in filesystem: {valid_filesystem_pairs}")
    print(f"[VERIFY] Total pairs recorded in manifest.json: {len(complete_pairs)}")

    if valid_filesystem_pairs == len(complete_pairs):
        print("[VERIFY SUCCESS] All complete pair folders are properly created and matched with manifest.json!")
    else:
        print("[VERIFY ERROR] Mismatch between filesystem structure and manifest!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
