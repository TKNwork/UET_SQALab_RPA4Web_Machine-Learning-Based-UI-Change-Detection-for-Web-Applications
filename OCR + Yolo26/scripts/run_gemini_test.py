#!/usr/bin/env python3
"""
Run Gemini Grounding Benchmark Test on WebClick Samples (0026 -> 0099: 74 samples).

Rules & Constraints:
1. Target Range: 0026 -> 0099 (74 samples).
2. Model: Fixed `gemini-3.6-flash`. No fallback.
3. Input: Manifest `webclick_vlm_input/manifest.jsonl`.
4. NO Ground Truth reading or evaluation.
5. Resume support: Skips valid predictions with model `gemini-3.6-flash`.
6. Rate limiting: Concurrency = 1 with sequential delay.
7. Crash safety: Immediate checkpoint write per sample.
8. Billing / Quota protection: Stops immediately on permanent API errors.
"""

import os
import sys
import json
import time
import argparse
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemini import (
    get_gemini_api_key,
    run_sample_grounding,
    PermanentAPIError,
    HAS_GENAI,
)

if HAS_GENAI:
    from google import genai


def main():
    parser = argparse.ArgumentParser(description="Run Gemini WebClick Benchmark (0026 -> 0099).")
    parser.add_argument("--model", type=str, default="gemini-3.6-flash", help="Gemini model identifier.")
    parser.add_argument("--manifest", type=str, default="webclick_vlm_input/manifest.jsonl", help="Manifest JSONL path.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum sample ID count (default: 100).")
    parser.add_argument("--start_id", type=int, default=26, help="Start sample ID (default: 26).")
    parser.add_argument("--end_id", type=int, default=99, help="End sample ID (default: 99).")
    parser.add_argument("--delay", type=float, default=13.0, help="Delay in seconds between API requests (default: 13.0 for 5 RPM free tier).")

    args = parser.parse_args()

    # Enforce strictly gemini-3.6-flash unless overridden intentionally
    target_model = args.model
    if target_model != "gemini-3.6-flash":
        print(f"[WARNING] Requested model is '{target_model}'. Standard benchmark requirement is 'gemini-3.6-flash'.")

    print("=" * 70)
    print(f"GEMINI WEBCLICK BENCHMARK INFERENCE (SAMPLES {args.start_id:04d} -> {args.end_id:04d})")
    print(f"Target Model: {target_model} | Limit: {args.limit} | Concurrency: 1 | Delay: {args.delay}s")
    print("=" * 70)


    # 1. Check API Key
    api_key, key_source = get_gemini_api_key()
    if not api_key:
        print("\n[ERROR] GEMINI_API_KEY not found in environment or config/geminiapi.txt.")
        sys.exit(1)

    masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "masked"
    print(f"API Key Source : {key_source}")
    print(f"API Key Masked : {masked_key}")

    if not HAS_GENAI:
        print("\n[ERROR] google-genai SDK not installed.")
        sys.exit(1)

    client = genai.Client(api_key=api_key) if HAS_GENAI and not api_key.startswith("sk-") else None

    # 2. Load Manifest & Filter Samples
    manifest_path = PROJECT_ROOT / args.manifest
    if not manifest_path.exists():
        print(f"\n[ERROR] Manifest not found at: {manifest_path}")
        sys.exit(1)

    all_records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

    # Filter strictly for sample_id >= start_id and sample_id <= end_id
    target_records = [r for r in all_records if args.start_id <= r["sample_id"] <= args.end_id]

    print(f"Manifest Total Records : {len(all_records)}")
    print(f"Filtered Target Range   : {len(target_records)} samples ({args.start_id:04d} -> {args.end_id:04d})\n")

    if not target_records:
        print("[ERROR] No records match the target range. Exiting.")
        sys.exit(0)

    predictions_dir = PROJECT_ROOT / "output_gemini_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # 3. Main Benchmark Loop
    results = []
    completed_new = 0
    skipped_existing = 0
    failed_count = 0
    total_target = len(target_records)

    permanent_error_occurred = None

    for idx, sample_rec in enumerate(target_records, 1):
        sid = sample_rec["sample_id"]
        sid_str = f"{sid:04d}"

        # Resume check preview log
        pred_file = predictions_dir / f"{sid_str}.json"
        is_already_valid = False
        if pred_file.exists():
            try:
                with open(pred_file, "r", encoding="utf-8") as pf:
                    pdata = json.load(pf)
                if pdata.get("valid") is True and pdata.get("model") == target_model:
                    is_already_valid = True
            except Exception:
                pass

        if is_already_valid:
            print(f"[SKIP] {sid_str} already valid (model={target_model})")
        else:
            print(f"[RUN ] {sid_str} | model={target_model}")

        try:
            res = run_sample_grounding(
                client=client,
                sample_record=sample_rec,
                model_name=target_model,
                predictions_dir=predictions_dir,
                max_retries=3
            )
            results.append(res)

            status = res.get("status", "unknown")
            valid = res.get("valid", False)
            lat = res.get("latency_ms", 0)
            pid = res.get("predicted_element_id")

            if status == "skipped":
                skipped_existing += 1
            elif valid:
                completed_new += 1
                print(f"[DONE] {sid_str} -> id={pid} | latency={lat} ms")
            else:
                failed_count += 1
                err = res.get("error", "Unknown error")
                print(f"[FAIL] {sid_str} -> status={status} | error={err[:60]}...")

            # Sequential pacing delay for non-skipped API requests to respect Free Tier 5 RPM limit
            if status != "skipped" and idx < total_target and args.delay > 0:
                time.sleep(args.delay)


        except PermanentAPIError as pe:
            print(f"\n[CRITICAL ERROR] Permanent API Error encountered on sample {sid_str}: {pe}")
            print("Stopping batch immediately to protect quota/billing.")
            permanent_error_occurred = str(pe)
            break
        except Exception as ex:
            print(f"\n[UNEXPECTED ERROR] Exception on sample {sid_str}: {ex}")
            failed_count += 1

    # 4. Post-Batch Audit & Classification
    print("\n" + "=" * 70)
    print(f"POST-BATCH INFERENCE AUDIT & CLASSIFICATION ({args.start_id:04d} -> {args.end_id:04d})")
    print("=" * 70)

    audit_valid = 0
    audit_failed = 0
    audit_missing = 0
    audit_corrupt = 0
    all_use_target_model = True

    invalid_responses_list = []
    api_failed_list = []
    input_failed_list = []
    successful_list = []

    prompt_tokens_list = []
    output_tokens_list = []
    total_tokens_list = []
    latencies_list = []

    for sid in range(args.start_id, args.end_id + 1):
        stem = f"{sid:04d}"
        pf = predictions_dir / f"{stem}.json"

        if not pf.exists():
            audit_missing += 1
            continue

        try:
            with open(pf, "r", encoding="utf-8") as f:
                data = json.load(f)

            model_used = data.get("model")
            if model_used != target_model:
                all_use_target_model = False

            is_valid = data.get("valid") is True
            pid = data.get("predicted_element_id")
            status = data.get("status")

            if is_valid and isinstance(pid, int):
                audit_valid += 1
                successful_list.append(data)
                lat = data.get("latency_ms", 0)
                if lat > 0:
                    latencies_list.append(lat)

                usage = data.get("usage")
                if usage and isinstance(usage, dict):
                    if usage.get("prompt_tokens") is not None:
                        prompt_tokens_list.append(usage["prompt_tokens"])
                    if usage.get("output_tokens") is not None:
                        output_tokens_list.append(usage["output_tokens"])
                    if usage.get("total_tokens") is not None:
                        total_tokens_list.append(usage["total_tokens"])
            else:
                audit_failed += 1
                if status == "invalid_response":
                    invalid_responses_list.append(data)
                elif status == "input_error":
                    input_failed_list.append(data)
                else:
                    api_failed_list.append(data)

        except Exception:
            audit_corrupt += 1

    # Latency Stats
    avg_latency = int(statistics.mean(latencies_list)) if latencies_list else 0
    median_latency = int(statistics.median(latencies_list)) if latencies_list else 0
    min_latency = min(latencies_list) if latencies_list else 0
    max_latency = max(latencies_list) if latencies_list else 0

    # Token Stats
    total_prompt_tokens = sum(prompt_tokens_list)
    total_output_tokens = sum(output_tokens_list)
    total_tokens = sum(total_tokens_list)

    avg_prompt_tokens = int(total_prompt_tokens / len(prompt_tokens_list)) if prompt_tokens_list else 0
    avg_output_tokens = int(total_output_tokens / len(output_tokens_list)) if output_tokens_list else 0

    # 5. Print Final Summary Report
    print(f"\n# Gemini WebClick {args.end_id - args.start_id + 1}-Sample Inference Summary\n")
    print(f"Range:")
    print(f"{args.start_id:04d} → {args.end_id:04d}\n")
    print(f"Target samples:")
    print(f"{args.end_id - args.start_id + 1}\n")
    print(f"Successful:")
    print(f"{audit_valid}\n")
    print(f"Invalid responses:")
    print(f"{len(invalid_responses_list)}\n")
    print(f"API failed:")
    print(f"{len(api_failed_list)}\n")
    print(f"Input failed:")
    print(f"{len(input_failed_list)}\n")
    print(f"Skipped existing:")
    print(f"{skipped_existing}\n")
    print(f"Average latency:")
    print(f"{avg_latency} ms\n")
    print(f"Median latency:")
    print(f"{median_latency} ms\n")
    print(f"Min latency:")
    print(f"{min_latency} ms\n")
    print(f"Max latency:")
    print(f"{max_latency} ms\n")

    print("## Token Usage Summary\n")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total tokens: {total_tokens}\n")
    print(f"Average prompt tokens/sample: {avg_prompt_tokens}")
    print(f"Average output tokens/sample: {avg_output_tokens}\n")

    print("## Classification Check\n")
    print(f"VALID: {audit_valid}")
    print(f"FAILED: {audit_failed}")
    print(f"MISSING: {audit_missing}")
    print(f"CORRUPT: {audit_corrupt}\n")

    print(f"All successful samples use {target_model}: {all_use_target_model}\n")

    # Error Table
    error_items = invalid_responses_list + api_failed_list + input_failed_list
    if error_items:
        print("## Error Table\n")
        print("| Sample ID | Status | Error |")
        print("| :--- | :--- | :--- |")
        for err_item in error_items:
            sid_s = f"{err_item['sample_id']:04d}"
            st = err_item.get("status", "failed")
            er_msg = str(err_item.get("error", "Unknown error"))
            if len(er_msg) > 60:
                er_msg = er_msg[:57] + "..."
            print(f"| {sid_s} | {st} | {er_msg} |")
        print()

    if permanent_error_occurred:
        print(f"[STOPPED] Inference stopped early due to permanent API error: {permanent_error_occurred}")
    else:
        print("=" * 70)
        print(f"INFERENCE COMPLETED FOR RANGE {args.start_id:04d} -> {args.end_id:04d}. RUNNER STOPPED.")
        print("=" * 70)


if __name__ == "__main__":
    main()
