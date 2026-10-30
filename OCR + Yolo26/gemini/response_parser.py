"""
Response Parser for Gemini Grounding Model Predictions.
"""

import json
import re


def parse_gemini_response(response_text: str, valid_element_ids: list[int]) -> dict:
    """
    Parses and validates Gemini API response.
    Expected JSON format: {"element_id": <integer>}

    Returns:
    {
        "is_valid": bool,
        "element_id": int | None,
        "error": str | None
    }
    """
    if not response_text or not isinstance(response_text, str):
        return {"is_valid": False, "element_id": None, "error": "Response text is empty or not string"}

    clean_text = response_text.strip()

    # Extract JSON if wrapped in markdown codeblock ```json ... ```
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
    if json_match:
        clean_text = json_match.group(1).strip()
    else:
        # Extract first JSON object if surrounded by prose
        brace_match = re.search(r"\{.*?\}", clean_text, re.DOTALL)
        if brace_match:
            clean_text = brace_match.group(0).strip()

    try:
        data = json.loads(clean_text)
    except Exception as e:
        return {"is_valid": False, "element_id": None, "error": f"JSON parse error: {e}"}

    if not isinstance(data, dict):
        return {"is_valid": False, "element_id": None, "error": "Parsed JSON is not an object"}

    if "element_id" not in data:
        return {"is_valid": False, "element_id": None, "error": "Missing 'element_id' key in JSON response"}

    elem_id = data["element_id"]

    if isinstance(elem_id, bool) or not isinstance(elem_id, int):
        return {"is_valid": False, "element_id": None, "error": f"'element_id' is not integer: {type(elem_id)} ({elem_id})"}

    if valid_element_ids is not None and elem_id not in valid_element_ids:
        return {
            "is_valid": False,
            "element_id": elem_id,
            "error": f"Invalid element_id {elem_id}: Not present in sample's element IDs list {min(valid_element_ids)}..{max(valid_element_ids)}"
        }

    return {"is_valid": True, "element_id": elem_id, "error": None}
