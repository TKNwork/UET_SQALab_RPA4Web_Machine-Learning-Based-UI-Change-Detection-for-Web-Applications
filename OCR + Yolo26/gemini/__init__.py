"""
Gemini Input Builder & Prompt Formatting Package for WebClick Benchmark.
"""

from .prompts import SYSTEM_GROUNDING_PROMPT, build_grounding_prompt, serialize_ui_elements
from .input_builder import load_instructions_map, validate_sample_files, build_sample_input, build_manifest
from .response_parser import parse_gemini_response
from .runner import get_gemini_api_key, get_proxy_url, run_sample_grounding, PermanentAPIError, HAS_GENAI

__all__ = [
    "SYSTEM_GROUNDING_PROMPT",
    "build_grounding_prompt",
    "serialize_ui_elements",
    "load_instructions_map",
    "validate_sample_files",
    "build_sample_input",
    "build_manifest",
    "parse_gemini_response",
    "get_gemini_api_key",
    "get_proxy_url",
    "run_sample_grounding",
    "PermanentAPIError",
    "HAS_GENAI",
]

