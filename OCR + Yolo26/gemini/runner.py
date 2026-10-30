"""
Gemini Grounding API Runner Module for WebClick Benchmark.

Sends multimodal requests (Original Screenshot + Annotated Screenshot + Structured Prompt Text)
to Gemini API and saves prediction outputs per sample.

STRICT SAFETY AUDIT: Ground truth is NEVER accessed.
"""

import os
import sys
import time
import json
import re
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from .prompts import build_grounding_prompt
from .response_parser import parse_gemini_response


import base64
import requests


class PermanentAPIError(Exception):
    """Raised when the Gemini API returns a non-retriable error (401, 403, billing/quota depleted)."""
    pass


def get_gemini_api_key() -> tuple[str | None, str]:
    """
    Retrieves GEMINI_API_KEY from environment variable first,
    or falls back to config/geminiapi.txt if available.
    Returns: (api_key_string or None, source_description)
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key, "environment variable GEMINI_API_KEY"

    key_file = PROJECT_ROOT / "config/geminiapi.txt"
    if key_file.exists():
        file_key = key_file.read_text(encoding='utf-8').strip()
        if file_key:
            return file_key, "config/geminiapi.txt"

    return None, "none"


def get_proxy_url() -> str | None:
    """Retrieves Proxy Base URL from env or config/proxy_url.txt."""
    url = os.environ.get("GEMINI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if url:
        return url.strip()
    file_path = PROJECT_ROOT / "config/proxy_url.txt"
    if file_path.exists():
        text = file_path.read_text(encoding='utf-8').strip()
        if text:
            return text
    return None


def call_openai_proxy(api_key: str, proxy_url: str, model_name: str, orig_path: Path, ann_path: Path, prompt_text: str) -> tuple[str, dict]:
    """Executes multimodal chat completion call via OpenAI-compatible Proxy URL."""
    with open(orig_path, "rb") as f:
        b64_orig = base64.b64encode(f.read()).decode("utf-8")
    with open(ann_path, "rb") as f:
        b64_ann = base64.b64encode(f.read()).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_orig}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_ann}"}},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0
    }

    url = f"{proxy_url.rstrip('/')}/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        err_body = resp.text
        if resp.status_code in (401, 403):
            raise PermanentAPIError(f"Permanent Proxy Auth Failure (HTTP {resp.status_code}): {err_body}")
        raise RuntimeError(f"Proxy HTTP {resp.status_code}: {err_body}")

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"Empty choices in proxy response: {data}")

    msg = choices[0].get("message", {})
    text = msg.get("content", "")

    usage = data.get("usage", {})
    usage_info = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens")
    }

    return text, usage_info


def run_sample_grounding(
    client,
    sample_record: dict,
    model_name: str = "gemini-3.6-flash",
    predictions_dir: Path = None,
    max_retries: int = 3
) -> dict:
    """
    Executes Gemini multimodal grounding inference for a single sample record.
    Saves prediction JSON to predictions_dir/<id:04d>.json.
    Supports resume (skips if valid prediction exists with exact model).
    """
    sid = sample_record["sample_id"]
    stem = f"{sid:04d}"

    if predictions_dir is None:
        predictions_dir = PROJECT_ROOT / "output_gemini_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    pred_file = predictions_dir / f"{stem}.json"

    # 1. Resume check
    if pred_file.exists():
        try:
            with open(pred_file, "r", encoding="utf-8") as f:
                existing_pred = json.load(f)
            if existing_pred.get("valid") is True and existing_pred.get("model") == model_name:
                existing_pred["status"] = "skipped"
                return existing_pred
        except Exception:
            pass

    # 2. Input consistency validation
    def _resolve_path(key1: str, key2: str) -> Path:
        val = sample_record.get(key1) or sample_record.get(key2) or ""
        if not val:
            return PROJECT_ROOT / ""
        p = PROJECT_ROOT / val
        if p.exists():
            return p
        p_vlm = PROJECT_ROOT / "webclick_vlm_input" / val
        if p_vlm.exists():
            return p_vlm
        return p

    orig_img_path = _resolve_path("image_path", "image")
    ann_img_path = _resolve_path("annotated_image_path", "annotated")
    compact_json_path = _resolve_path("compact_json_path", "compact")
    instruction_text = sample_record.get("instruction", "")

    input_errors = []
    if not orig_img_path.exists():
        input_errors.append(f"Original image missing: {orig_img_path}")
    if not ann_img_path.exists():
        input_errors.append(f"Annotated image missing: {ann_img_path}")
    if not compact_json_path.exists():
        input_errors.append(f"Compact JSON missing: {compact_json_path}")
    if not instruction_text or not isinstance(instruction_text, str):
        input_errors.append("Instruction missing or invalid")

    compact_data = None
    candidate_ids = []
    if compact_json_path.exists():
        try:
            with open(compact_json_path, "r", encoding="utf-8") as f:
                compact_data = json.load(f)
            elements = compact_data.get("elements", [])
            candidate_ids = [e["element_id"] for e in elements if isinstance(e, dict) and "element_id" in e]
            
            # Check compact_data image match
            img_field = str(compact_data.get("image", ""))
            img_stem = Path(img_field).stem if img_field else ""
            if img_stem and img_stem not in (str(sid), stem):
                input_errors.append(f"Compact JSON image mismatch: field '{img_field}' vs sample {sid}")
        except Exception as e:
            input_errors.append(f"Compact JSON load failure: {e}")

    if input_errors:
        err_msg = "; ".join(input_errors)
        result = {
            "sample_id": sid,
            "model": model_name,
            "instruction": instruction_text,
            "predicted_element_id": None,
            "valid": False,
            "latency_ms": 0,
            "raw_response": None,
            "error": err_msg,
            "usage": None,
            "status": "input_error"
        }
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    # 3. Build prompt text & load images
    elements = compact_data.get("elements", [])
    prompt_text = build_grounding_prompt(instruction_text, elements)

    try:
        orig_img = Image.open(orig_img_path).convert("RGB")
        ann_img = Image.open(ann_img_path).convert("RGB")
    except Exception as e:
        err_msg = f"Failed to open image files for sample {stem}: {e}"
        result = {
            "sample_id": sid,
            "model": model_name,
            "instruction": instruction_text,
            "predicted_element_id": None,
            "valid": False,
            "latency_ms": 0,
            "raw_response": None,
            "error": err_msg,
            "usage": None,
            "status": "input_error"
        }
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    # 4. Multimodal payload & schema config
    contents = [orig_img, ann_img, prompt_text]

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "OBJECT",
            "properties": {
                "element_id": {"type": "INTEGER"}
            },
            "required": ["element_id"]
        },
        temperature=0.0
    )

    # 5. Retry loop (max_retries = 3 attempts)
    last_exception = None
    raw_response_text = None
    latency_ms = 0
    usage_info = None

    delays = [0.0, 2.0, 4.0, 8.0]

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            backoff_sleep = delays[min(attempt - 1, len(delays) - 1)]
            time.sleep(backoff_sleep)

        try:
            start_time = time.perf_counter()
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            if hasattr(response, "text") and response.text:
                raw_response_text = response.text
            else:
                raw_response_text = str(response)

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage_info = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", None),
                    "output_tokens": getattr(response.usage_metadata, "candidates_token_count", None),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", None)
                }

            break
        except Exception as e:
            last_exception = e
            err_str = str(e)
            err_lower = err_str.lower()

            # Check for Rate Limit (429 / RESOURCE_EXHAUSTED)
            is_429 = "429" in err_str or "resource_exhausted" in err_lower or "quota" in err_lower or "rate limit" in err_lower

            if is_429:
                # 429 rate limit is transient. Sleep 15s to clear 5 RPM window.
                if attempt < max_retries:
                    print(f"[RETRY {attempt}/{max_retries}] {stem} | model={model_name} -> 429 Rate Limit (Sleeping 15s)...")
                    time.sleep(15.0)
                    continue

            # Permanent / Unrecoverable billing account or auth errors
            permanent_keywords = [
                "401", "403", "unauthorized",
                "permission_denied", "account disabled", "quota exhausted permanently",
                "prepayment credits are depleted", "billing account disabled"
            ]
            if not is_429 and any(kw in err_lower for kw in permanent_keywords):
                raise PermanentAPIError(f"Permanent Gemini API failure on sample {stem}: {err_str}")

            if attempt < max_retries:
                print(f"[RETRY {attempt}/{max_retries}] {stem} | model={model_name} -> {err_str[:60]}...")


    if raw_response_text is None:
        err_msg = f"API call failed after {max_retries} attempts: {last_exception}"
        result = {
            "sample_id": sid,
            "model": model_name,
            "instruction": instruction_text,
            "predicted_element_id": None,
            "valid": False,
            "latency_ms": latency_ms,
            "raw_response": None,
            "error": err_msg,
            "usage": None,
            "status": "failed"
        }
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    # 6. Response validation
    parsed_res = parse_gemini_response(raw_response_text, candidate_ids)

    result = {
        "sample_id": sid,
        "model": model_name,
        "instruction": instruction_text,
        "predicted_element_id": parsed_res["element_id"],
        "valid": parsed_res["is_valid"],
        "latency_ms": latency_ms,
        "raw_response": raw_response_text,
        "error": parsed_res["error"],
        "usage": usage_info,
        "status": "done" if parsed_res["is_valid"] else "invalid_response"
    }

    # 7. Checkpoint / Crash safety: Save prediction file immediately
    with open(pred_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result

