"""Extract OCR evidence from fused hierarchy detections without changing the source JSON."""
import json
from pathlib import Path

from shared.matching import GENERIC_LABELS, normalize_detection, normalize_text


def hierarchy_ocr_text(element):
    source = element.get('source')
    sources = element.get('sources')
    has_ocr_provenance = source == 'ocr' or (isinstance(sources, list) and 'ocr' in sources)
    if not has_ocr_provenance:
        return ''
    text = element.get('text')
    if normalize_text(text) not in GENERIC_LABELS:
        return text
    # Legacy OCR rows store the transcription as their class label. A YOLO
    # class label is never a substitute for missing OCR text.
    label = element.get('class_label')
    if source == 'ocr' and normalize_text(label) not in GENERIC_LABELS:
        return label
    return ''


def load_hierarchy_text(path):
    """Build the text matcher's input view while retaining fused-box provenance."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return normalize_hierarchy_text(data, path)


def normalize_hierarchy_text(data, path):
    elements = data.get('elements')
    if not isinstance(elements, list):
        raise ValueError(f'Expected an elements list in {path}')
    detections, warnings = [], []
    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            warnings.append(f'Invalid element at index {index}')
            continue
        text = hierarchy_ocr_text(element)
        if not text:
            continue
        element_id = element.get('id', f'__idx_{index}')
        try:
            detection = normalize_detection(
                {**element, 'source': 'ocr', 'class_label': text}, element_id)
        except (TypeError, ValueError) as error:
            warnings.append(str(error))
            continue
        detection.update(hierarchy_source=element.get('source'),
                         hierarchy_sources=element.get('sources'),
                         hierarchy_class_label=element.get('class_label'))
        detections.append(detection)
    if len({element['id'] for element in detections}) != len(detections):
        raise ValueError(f'Duplicate text detection IDs in {path}')
    return detections, warnings
