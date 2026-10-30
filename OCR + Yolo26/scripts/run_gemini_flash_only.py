#!/usr/bin/env python3
"""
Pure VLM Gemini 3.6 Flash Grounding Runner (Standalone Baseline).

Sends ONLY the Original Screenshot + Instruction to Gemini 3.6 Flash.
Predicts normalized bounding box [ymin, xmin, ymax, xmax] (scale 0-1000).
Saves outputs to output_gemini_flash_only_predictions/<stem>.json.
"""

import os
import sys
import time
import json
import base64
import argparse
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemini import get_gemini_api_key, get_proxy_url, PermanentAPIError, load_instructions_map

PURE_VLM_SYSTEM_PROMPT = """You are a GUI visual grounding assistant.
Given a webpage screenshot and a user instruction, identify the exact target UI element to interact with and determine its center click point.

Output the click point coordinates as [x, y] in normalized coordinates on a scale of 0 to 1000, where:
- x is the horizontal position from left (0) to right (1000).
- y is the vertical position from top (0) to bottom (1000).

Rules:
- Target the precise center of the target interactive element.
- Do not output any text or explanation.
- Return ONLY valid JSON in the format:
{
  "point": [x, y]
}"""


def build_pure_vlm_prompt(instruction: str) -> str:
    return (
        f"{PURE_VLM_SYSTEM_PROMPT}\n\n"
        f"Instruction:\n{instruction}"
    )


def run_pure_vlm_sample(
    api_key: str,
    proxy_url: str,
    sample_id: int,
    image_path: Path,
    instruction: str,
    model_name: str = "gemini-3.6-flash",
    output_dir: Path = None,
    max_retries: int = 3
) -> dict:
    stem = f"{sample_id:04d}"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output_gemini_flash_only_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_file = output_dir / f"{stem}.json"
    if pred_file.exists():
        try:
            with open(pred_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("valid") is True and existing.get("model") == model_name:
                existing["status"] = "skipped"
                return existing
        except Exception:
            pass

    if not image_path.exists():
        res = {
            "sample_id": sample_id,
            "model": model_name,
            "instruction": instruction,
            "predicted_bbox": None,
            "valid": False,
            "latency_ms": 0,
            "error": f"Original image not found: {image_path}",
            "status": "input_error"
        }
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        return res

    with open(image_path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")

    prompt_text = build_pure_vlm_prompt(instruction)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    url = f"{proxy_url.rstrip('/')}/chat/completions"

    raw_response_text = None
    latency_ms = 0
    usage_info = None
    last_err = None

    delays = [0.0, 2.0, 4.0, 8.0]

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            time.sleep(delays[min(attempt - 1, len(delays) - 1)])

        try:
            start_t = time.perf_counter()
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            latency_ms = int((time.perf_counter() - start_t) * 1000)

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    raw_response_text = choices[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                usage_info = {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens")
                }
                break
            else:
                err_text = resp.text
                if resp.status_code in (401, 403):
                    raise PermanentAPIError(f"Permanent Proxy Auth Failure (HTTP {resp.status_code}): {err_text}")
                last_err = f"HTTP {resp.status_code}: {err_text}"
        except PermanentAPIError:
            raise
        except Exception as e:
            last_err = str(e)

    if raw_response_text is None:
        res = {
            "sample_id": sample_id,
            "model": model_name,
            "instruction": instruction,
            "predicted_point": None,
            "valid": False,
            "latency_ms": latency_ms,
            "raw_response": None,
            "error": last_err,
            "usage": None,
            "status": "failed"
        }
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        return res

    # Parse JSON & point [x, y]
    parsed_point = None
    is_valid = False
    parse_error = None

    clean_text = raw_response_text.strip()
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
    if m:
        clean_text = m.group(1).strip()
    else:
        m2 = re.search(r"\{.*?\}", clean_text, re.DOTALL)
        if m2:
            clean_text = m2.group(0).strip()

    try:
        data = json.loads(clean_text)
        pt = data.get("point") or data.get("click_point") or data.get("point_to_click")
        if isinstance(pt, list) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt):
            parsed_point = [float(pt[0]), float(pt[1])]
            is_valid = True
        elif "bbox" in data and isinstance(data["bbox"], list) and len(data["bbox"]) == 4:
            # Fallback: if model returns bbox [ymin, xmin, ymax, xmax], compute center [cx, cy]
            ymin, xmin, ymax, xmax = [float(v) for v in data["bbox"]]
            parsed_point = [round((xmin + xmax) / 2.0, 2), round((ymin + ymax) / 2.0, 2)]
            is_valid = True
        else:
            parse_error = f"Invalid point format in JSON: {data}"
    except Exception as e:
        parse_error = f"JSON parse error: {e}"

    res = {
        "sample_id": sample_id,
        "model": model_name,
        "instruction": instruction,
        "predicted_point": parsed_point,
        "valid": is_valid,
        "latency_ms": latency_ms,
        "raw_response": raw_response_text,
        "error": parse_error,
        "usage": usage_info,
        "status": "done" if is_valid else "invalid_response"
    }

    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    return res


def main():
    parser = argparse.ArgumentParser(description="Run Pure VLM Gemini 3.6 Flash Grounding Baseline.")
    parser.add_argument("--model", type=str, default="gemini-3.6-flash", help="Model identifier.")
    parser.add_argument("--start_id", type=int, default=0, help="Start sample ID.")
    parser.add_argument("--end_id", type=int, default=99, help="End sample ID (default 99 for 100 samples).")
    parser.add_argument("--delay", type=float, default=2.0, help="Sequential delay between API requests in seconds.")
    parser.add_argument("--output_dir", type=str, default="output_gemini_flash_only_predictions", help="Output directory name.")

    args = parser.parse_args()

    api_key, key_source = get_gemini_api_key()
    if not api_key:
        print("[ERROR] API Key missing. Set config/geminiapi.txt or GEMINI_API_KEY")
        sys.exit(1)

    proxy_url = get_proxy_url() or "https://cashop.io.vn/v1"
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    instructions_path = PROJECT_ROOT / "data/webclick/instructions.json"
    instructions_map = load_instructions_map(instructions_path)
    images_dir = PROJECT_ROOT / "data/webclick/images"

    target_sids = [sid for sid in sorted(instructions_map.keys()) if args.start_id <= sid <= args.end_id]

    print("=" * 70)
    print(f"PURE VLM GEMINI 3.6 FLASH BASELINE RUNNER (SAMPLES {args.start_id:04d} -> {args.end_id:04d})")
    print(f"Target Model       : {args.model}")
    print(f"Proxy URL          : {proxy_url}")
    print(f"Output Directory   : {output_dir.relative_to(PROJECT_ROOT)}/")
    print(f"Total Target Count : {len(target_sids)} samples")
    print("=" * 70)

    results = []
    completed_new = 0
    skipped_existing = 0
    failed_count = 0

    for idx, sid in enumerate(target_sids, 1):
        stem = f"{sid:04d}"
        img_path = images_dir / f"{stem}.png"
        instr_text = instructions_map[sid]["instruction"]

        pred_file = output_dir / f"{stem}.json"
        is_already_valid = False
        if pred_file.exists():
            try:
                with open(pred_file, "r", encoding="utf-8") as pf:
                    pdata = json.load(pf)
                if pdata.get("valid") is True and pdata.get("model") == args.model and pdata.get("predicted_point") is not None:
                    is_already_valid = True
            except Exception:
                pass

        if is_already_valid:
            print(f"[SKIP] {stem} already valid")
            skipped_existing += 1
            continue

        print(f"[RUN ] {stem} | model={args.model}")

        try:
            res = run_pure_vlm_sample(
                api_key=api_key,
                proxy_url=proxy_url,
                sample_id=sid,
                image_path=img_path,
                instruction=instr_text,
                model_name=args.model,
                output_dir=output_dir,
                max_retries=3
            )
            results.append(res)

            status = res.get("status")
            valid = res.get("valid", False)
            lat = res.get("latency_ms", 0)
            pt = res.get("predicted_point")

            if status == "skipped":
                skipped_existing += 1
            elif valid:
                completed_new += 1
                print(f"[DONE] {stem} -> point={pt} | latency={lat} ms")
            else:
                failed_count += 1
                err = res.get("error", "Unknown error")
                print(f"[FAIL] {stem} -> status={status} | error={err[:60]}...")

            if status != "skipped" and idx < len(target_sids) and args.delay > 0:
                time.sleep(args.delay)

        except PermanentAPIError as pe:
            print(f"\n[CRITICAL ERROR] Permanent API error on {stem}: {pe}")
            print("Stopping batch run to protect quota.")
            break
        except Exception as ex:
            print(f"\n[UNEXPECTED ERROR] Exception on {stem}: {ex}")
            failed_count += 1

    print("\n" + "=" * 70)
    print("PURE VLM GEMINI 3.6 FLASH RUN COMPLETE")
    print(f"Processed: {completed_new}, Skipped: {skipped_existing}, Failed: {failed_count}")
    print("=" * 70)


if __name__ == "__main__":
    main()
