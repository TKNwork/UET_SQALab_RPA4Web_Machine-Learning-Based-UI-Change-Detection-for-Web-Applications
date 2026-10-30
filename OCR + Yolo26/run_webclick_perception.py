#!/usr/bin/env python3
"""
Robust Batch WebClick Perception Pipeline Runner with Resume & Validation.
Scans data/webclick/images/ (1,639 samples: 0000.png -> 1638.png)
Gens: output_webclick_perception/<stem>/<stem>_ui.json & <stem>_ui_compact.json
"""

import os
import sys
import time
import json
import datetime
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_perception import process_perception


def validate_sample_completion(sample_out_dir: Path, stem: str) -> bool:
    """
    Kiểm tra một sample đã hoàn thành đầy đủ và hợp lệ 2 file JSON hay chưa:
    1. <stem>_ui.json
    2. <stem>_ui_compact.json
    """
    full_path = sample_out_dir / f"{stem}_ui.json"
    compact_path = sample_out_dir / f"{stem}_ui_compact.json"

    if not full_path.exists() or not compact_path.exists():
        return False

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)
        with open(compact_path, "r", encoding="utf-8") as f:
            compact_data = json.load(f)

        # Validate cấu trúc compact_data
        if not isinstance(compact_data, dict):
            return False

        required_keys = ["image", "width", "height", "elements"]
        for k in required_keys:
            if k not in compact_data:
                return False

        elements = compact_data["elements"]
        if not isinstance(elements, list):
            return False

        # Validate từng element trong compact_data
        img_w = compact_data.get("width", 0)
        img_h = compact_data.get("height", 0)

        for elem in elements:
            if not isinstance(elem, dict):
                return False
            if "element_id" not in elem or "bbox" not in elem:
                return False
            if not isinstance(elem["element_id"], int):
                return False
            bbox = elem["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                return False
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h):
                log_file = sample_out_dir.parent / "resume_errors.log"
                log_resume_error(log_file, stem, "bbox_validation", Exception(f"Invalid bbox: elem_id={elem.get('element_id')}, bbox={bbox}, img_dim=({img_w}x{img_h})"))
                return False

        # Validate element_id uniqueness và continuity
        ids = [e["element_id"] for e in elements]
        if len(ids) != len(set(ids)) or sorted(ids) != list(range(len(elements))):
            return False

        return True
    except Exception:
        return False


def log_resume_error(log_file: Path, stem: str, stage: str, exc: Exception):
    """Ghi log lỗi ra file output_webclick_perception/resume_errors.log"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().isoformat()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] | Sample: {stem} | Stage: {stage} | Error: {str(exc)}\n")


def process_sample_with_retry(img_path: Path, output_dir: Path, visualize: bool, save_debug: bool, max_retries: int = 3) -> bool:
    """Xử lý 1 sample với tối đa 3 lần retry cho lỗi mạng/API"""
    stem = img_path.stem
    error_log_file = output_dir / "resume_errors.log"

    for attempt in range(1, max_retries + 1):
        try:
            # Im lặng không in log chi tiết từng bước perception trong batch để giữ console sạch
            process_perception(img_path, output_dir, visualize=visualize, save_debug=save_debug)
            
            # Kiểm tra lại tính hợp lệ sau khi xuất JSON
            sample_out_dir = output_dir / stem
            if validate_sample_completion(sample_out_dir, stem):
                return True
            else:
                raise ValueError("Generated JSON failed post-validation check.")
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.0 * attempt)  # Backoff ngắn giữa các lần retry
            else:
                log_resume_error(error_log_file, stem, f"attempt_{attempt}", e)
                return False
    return False


def run_batch_pass(image_files: list, output_dir: Path, visualize: bool, save_debug: bool, pass_num: int, stop_at_sample: str = "1145"):
    total = len(image_files)
    processed_new = 0
    skipped_existing = 0
    failed = 0

    print(f"\n============================================================")
    print(f"STARTING REVERSE PASS {pass_num} FOR {total} SAMPLE(S) (STOP AT {stop_at_sample})")
    print(f"============================================================")

    for idx, img_path in enumerate(image_files, 1):
        stem = img_path.stem
        sample_out_dir = output_dir / stem

        # Kiểm tra hardcode stop عند reaching sample 1145
        if stop_at_sample and (stem == stop_at_sample or (stem.isdigit() and int(stem) <= int(stop_at_sample))):
            if validate_sample_completion(sample_out_dir, stem):
                print(f"[STOP] Reached target sample {stem} (<= {stop_at_sample}) which is already complete. Stopping batch run.")
                skipped_existing += 1
                break
            else:
                print(f"[RUN ] {stem} (Attempting perception for target stop sample {stop_at_sample}...)")
                success = process_sample_with_retry(img_path, output_dir, visualize=visualize, save_debug=save_debug, max_retries=3)
                if success:
                    compact_file = sample_out_dir / f"{stem}_ui_compact.json"
                    try:
                        with open(compact_file, "r", encoding="utf-8") as f:
                            elem_count = len(json.load(f).get("elements", []))
                    except Exception:
                        elem_count = 0
                    print(f"[DONE & STOP] Reached and completed target sample {stem}. Stopping batch run.")
                    processed_new += 1
                else:
                    print(f"[FAIL & STOP] Target sample {stem} perception failed. Stopping batch run.")
                    failed += 1
                break

        if validate_sample_completion(sample_out_dir, stem):
            print(f"[SKIP] {stem} already complete")
            skipped_existing += 1
            continue

        print(f"[RUN ] {stem} (Attempting perception...)")
        success = process_sample_with_retry(img_path, output_dir, visualize=visualize, save_debug=save_debug, max_retries=3)

        if success:
            compact_file = sample_out_dir / f"{stem}_ui_compact.json"
            try:
                with open(compact_file, "r", encoding="utf-8") as f:
                    elem_count = len(json.load(f).get("elements", []))
            except Exception:
                elem_count = 0
            print(f"[DONE] {stem} - {elem_count} elements")
            processed_new += 1
        else:
            print(f"[FAIL] {stem} - perception error (Logged to resume_errors.log)")
            failed += 1

    remaining = total - (skipped_existing + processed_new)
    print(f"\nPass {pass_num} finished. Processed: {processed_new}, Skipped: {skipped_existing}, Failed: {failed}, Remaining: {remaining}")
    return skipped_existing + processed_new, failed


def main():
    parser = argparse.ArgumentParser(description="Robust Batch WebClick Perception Runner.")
    parser.add_argument("--input", type=str, default="data/webclick/images", help="Input directory containing screenshots.")
    parser.add_argument("--output", type=str, default="output_webclick_perception", help="Output base directory.")
    parser.add_argument("--visualize", action="store_true", help="Generate visualized debug images.")
    parser.add_argument("--save-debug", action="store_true", help="Save intermediate raw nodes & edges debug files.")
    parser.add_argument("--stop-at", type=str, default="1145", help="Hardcoded sample ID to stop at when reached in reverse order.")

    args = parser.parse_args()

    input_dir = PROJECT_ROOT / args.input
    output_dir = PROJECT_ROOT / args.output
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff')
    # Sắp xếp NGƯỢC LẠI từ 1638.png về 0000.png
    image_files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts], key=lambda x: x.name, reverse=True)

    total_samples = len(image_files)

    if total_samples == 0:
        print("=" * 60)
        print("WEBCLICK BATCH PERCEPTION")
        print("=" * 60)
        print("No WebClick images found.")
        print(f"Please place screenshots in: {input_dir}/")
        print("=" * 60)
        sys.exit(0)

    print("=" * 60)
    print("WEBCLICK BATCH PERCEPTION REVERSE RESUME RUNNER")
    print("=" * 60)
    print(f"Total dataset images found: {total_samples}")
    print(f"Execution order           : REVERSE (from {image_files[0].name} down to {image_files[-1].name})")
    print(f"Hardcoded stop target     : Sample {args.stop_at}")
    print(f"Output directory          : {output_dir}/\n")

    # Pass 1: Xử lý hoặc Skip theo thứ tự ngược
    run_batch_pass(image_files, output_dir, visualize=args.visualize, save_debug=args.save_debug, pass_num=1, stop_at_sample=args.stop_at)

    # Pass 2: Tự động retry riêng các sample còn thiếu / lỗi (nếu có) trong dải từ 1638 -> stop_at
    stop_idx = int(args.stop_at) if args.stop_at.isdigit() else 1145
    remaining_files = [f for f in image_files if (f.stem.isdigit() and int(f.stem) >= stop_idx) and not validate_sample_completion(output_dir / f.stem, f.stem)]
    if remaining_files:
        print(f"\nFound {len(remaining_files)} remaining/failed sample(s) in range >= {stop_idx}. Starting Pass 2 auto-retry...")
        run_batch_pass(remaining_files, output_dir, visualize=args.visualize, save_debug=args.save_debug, pass_num=2, stop_at_sample=args.stop_at)

    # Quét kiểm tra tổng kết cuối cùng cho toàn bộ 1,639 samples (0000 -> 1638)
    complete_count = 0
    missing_count = 0
    corrupt_count = 0
    failed_samples = []

    # Quét từ 0000 -> 1638 theo thứ tự chuẩn để report
    all_forward_files = sorted(image_files, key=lambda x: x.name)

    for f in all_forward_files:
        stem = f.stem
        s_dir = output_dir / stem
        full_p = s_dir / f"{stem}_ui.json"
        comp_p = s_dir / f"{stem}_ui_compact.json"

        if not full_p.exists() and not comp_p.exists():
            missing_count += 1
            failed_samples.append(stem)
        elif not validate_sample_completion(s_dir, stem):
            corrupt_count += 1
            failed_samples.append(stem)
        else:
            complete_count += 1

    print("\n" + "=" * 60)
    print("WEBCLICK PERCEPTION BATCH SUMMARY")
    print("=" * 60)
    print(f"Total samples : {total_samples}")
    print(f"Complete      : {complete_count}")
    print(f"Missing       : {missing_count}")
    print(f"Corrupt       : {corrupt_count}")
    print(f"Failed        : {len(failed_samples)}")

    if failed_samples:
        print(f"Failed sample IDs: {failed_samples[:20]}{'...' if len(failed_samples) > 20 else ''}")
        print(f"Check error details in: {output_dir}/resume_errors.log")
    print("=" * 60)


if __name__ == "__main__":
    main()

