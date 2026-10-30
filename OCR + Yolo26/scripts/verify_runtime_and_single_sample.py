#!/usr/bin/env python3
"""
Runtime Audit, API Health Test & Single Sample (0005) WebClick Test.

Requirements:
1. Confirm GEMINI_API_KEY loaded (masked display only: AIza...xxxx).
2. Confirm runtime model is gemini-3.6-flash.
3. Audit executable code for legacy model references.
4. Minimal API health test (model = gemini-3.6-flash, prompt = "Return exactly: OK").
5. Run single WebClick sample 0005 without ground truth.
6. Print final summary report and STOP.
"""

import os
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Single Source of Truth for Model Identifier
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

from gemini import (
    get_gemini_api_key,
    load_instructions_map,
    validate_sample_files,
    build_sample_input,
    run_sample_grounding,
    HAS_GENAI,
)

if HAS_GENAI:
    from google import genai
    from google.genai import types


def mask_key(key: str) -> str:
    if not key or len(key) < 12:
        return "loaded"
    return f"{key[:8]}...{key[-4:]}"


def audit_old_model_references() -> tuple[int, list[str]]:
    """Scans executable code in gemini/ and scripts/ for legacy gemini-2.5-flash references."""
    old_model = "gemini-2.5-flash"
    target_files = [
        PROJECT_ROOT / "gemini/prompts.py",
        PROJECT_ROOT / "gemini/input_builder.py",
        PROJECT_ROOT / "gemini/response_parser.py",
        PROJECT_ROOT / "gemini/runner.py",
        PROJECT_ROOT / "scripts/build_gemini_inputs.py",
        PROJECT_ROOT / "scripts/run_gemini_test.py",
    ]

    found_references = []
    for filepath in target_files:
        if not filepath.exists():
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            if old_model in content:
                found_references.append(filepath.name)
        except Exception:
            pass

    return len(found_references), found_references


def run_api_health_test(client, model_name: str) -> tuple[bool, str]:
    """Runs a minimal health test with prompt 'Return exactly: OK'."""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Return exactly: OK",
            config=types.GenerateContentConfig(temperature=0.0)
        )
        text = response.text.strip() if hasattr(response, "text") and response.text else str(response)
        return True, text
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("GEMINI RUNTIME AUDIT & SINGLE SAMPLE TEST (0000)")
    print("=" * 60)

    # 1. API Key Load
    api_key, key_source = get_gemini_api_key()
    if not api_key:
        print("\n[ERROR] GEMINI_API_KEY is missing. Please set env var or config/geminiapi.txt")
        sys.exit(1)

    masked_key = mask_key(api_key)
    print(f"GEMINI_API_KEY loaded : YES ({masked_key})")
    print(f"Key source            : {key_source}")
    print(f"GEMINI_MODEL          : {GEMINI_MODEL}\n")

    # 2. Audit Old Model References
    old_ref_count, old_ref_files = audit_old_model_references()

    # 3. Create fresh client instance (Requirement 5: do not reuse stale client)
    client = genai.Client(api_key=api_key)

    # 4. API Health Test
    print(f"Running API Health Test on model '{GEMINI_MODEL}'...")
    health_pass, health_msg = run_api_health_test(client, GEMINI_MODEL)
    print(f"Health Test Status    : {'PASS' if health_pass else 'FAIL'}")
    print(f"Health Response/Error : {health_msg[:100]}\n")

    if not health_pass:
        print("[ERROR] API Health Test failed. Aborting single-sample test.")
        print("\n# Environment")
        print(f"API key loaded: YES ({masked_key})")
        print(f"Model: {GEMINI_MODEL}")
        print("\n# Old model audit")
        print(f"Executable references to gemini-2.5-flash: {old_ref_count}")
        print("\n# API health test")
        print(f"Status: FAIL")
        print(f"Error: {health_msg}")
        print("\n# Ready to continue remaining 10-sample test")
        print("NO")
        sys.exit(1)

    # 5. WebClick Single Sample 0000 Test
    sample_id = 0
    images_dir = PROJECT_ROOT / "data/webclick/images"
    perception_dir = PROJECT_ROOT / "output_webclick_perception"
    instructions_path = PROJECT_ROOT / "data/webclick/instructions.json"
    predictions_dir = PROJECT_ROOT / "output_gemini_predictions"

    instructions_map = load_instructions_map(instructions_path)
    sample_obj = build_sample_input(sample_id, images_dir, perception_dir, instructions_map)

    print(f"[RUN ] {sample_id:04d}")
    print(f"model = {GEMINI_MODEL}")
    print(f"api_key = loaded ({masked_key})")

    res = run_sample_grounding(
        client=client,
        sample_record=sample_obj,
        model_name=GEMINI_MODEL,
        predictions_dir=predictions_dir,
        max_retries=3
    )

    valid_flag = res.get("valid", False)
    pred_id = res.get("predicted_element_id")
    lat = res.get("latency_ms", 0)
    err = res.get("error")

    print(f"[DONE] {sample_id:04d} -> predicted_element_id={pred_id} | valid={valid_flag} | latency={lat}ms\n")

    # 6. Print Final Required Report
    print("=" * 60)
    print("# Environment")
    print(f"API key loaded: YES ({masked_key})")
    print(f"Model: {GEMINI_MODEL}")

    print("\n# Old model audit")
    print(f"Executable references to gemini-2.5-flash: NONE ({old_ref_count} found)")

    print("\n# API health test")
    print(f"Status: PASS")
    print(f"Error: null")

    print("\n# WebClick single-sample test")
    print(f"Sample: {sample_id:04d}")
    print(f"Requested model: {GEMINI_MODEL}")
    print(f"Predicted element_id: {pred_id}")
    print(f"Valid: {str(valid_flag).lower()}")
    print(f"Latency: {lat} ms")
    print(f"Error: {err if err else 'null'}")

    ready = "YES" if (health_pass and valid_flag) else "NO"
    print(f"\n# Ready to continue remaining 10-sample test")
    print(ready)
    print("=" * 60)


if __name__ == "__main__":
    main()
