"""
Grounding System Prompts & Element Serialization for Gemini Pro GUI Benchmark.
"""

SYSTEM_GROUNDING_PROMPT = """You are a GUI grounding model.

You will receive:
1. The original webpage screenshot.
2. An annotated version of the same screenshot where each detected UI element is marked with a numeric element ID badge.
3. A structured list of detected UI elements.
4. A user instruction.

Your task is to identify the single UI element that should be clicked to accomplish the instruction.

Use both:
- visual information from the original screenshot,
- numbered boxes from the annotated screenshot,
- structured UI element information.

Each detected element has:
- element_id
- text
- class_label
- bbox
- parent

Rules:
- Select exactly one element from the provided element list.
- Never invent an element ID.
- The selected element_id must exist in the provided UI data.
- Do not output coordinates.
- Do not explain your reasoning.
- Return only the required JSON.

Output format:
{
  "element_id": <integer>
}"""


def serialize_ui_elements(elements: list) -> str:
    """
    Serializes list of compact UI element dicts into a token-efficient text format.
    Format:
    [0] class=divider | text=null | bbox=[1081.0, 748.0, 1093.0, 992.0] | parent=null
    [1] class=ui_element | text="Learn" | bbox=[508.7, 32.88, 550.74, 45.3] | parent=null
    """
    lines = []
    for elem in elements:
        eid = elem.get("element_id")
        class_label = elem.get("class_label", "unknown")
        raw_text = elem.get("text")
        bbox = elem.get("bbox", [])
        parent = elem.get("parent")

        if raw_text is None:
            text_str = "null"
        else:
            clean_text = str(raw_text).replace('\n', ' ').strip()
            text_str = f'"{clean_text}"'

        parent_str = "null" if parent is None else str(parent)

        line = f'[{eid}] class={class_label} | text={text_str} | bbox={bbox} | parent={parent_str}'
        lines.append(line)

    return "\n".join(lines)


def build_grounding_prompt(instruction: str, elements: list) -> str:
    """
    Assembles complete grounding prompt containing system instructions,
    user task instruction, and serialized UI elements.
    """
    serialized_elements = serialize_ui_elements(elements)
    
    full_prompt = (
        f"{SYSTEM_GROUNDING_PROMPT}\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Detected UI elements:\n{serialized_elements}"
    )
    return full_prompt
