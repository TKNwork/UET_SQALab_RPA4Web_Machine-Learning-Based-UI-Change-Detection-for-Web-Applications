import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from combine._core.relation_case import RelationCase
from combine._core.geometry import to_xyxy, compute_area, validate_box
from combine._core.matrix import compute_all_matrices
from combine._core.schema import load_ocr, load_small, load_medium, load_large, load_cv


def test_core_module():
    print("=== TESTING COMBINE _CORE MODULES ===")

    # 1. Test relation_case
    assert RelationCase.DUPLICATE.value == "duplicate"
    assert RelationCase.A_NESTED_IN_B.value == "a_nested_in_b"
    assert RelationCase.B_NESTED_IN_A.value == "b_nested_in_a"
    assert RelationCase.PARTIAL_OVERLAP.value == "partial_overlap"
    assert RelationCase.STANDALONE.value == "standalone"
    print("-> RelationCase Enum: OK")

    # 2. Test geometry
    converted_bbox = to_xyxy([10, 20, 30, 40], fmt='xywh')
    assert converted_bbox == [10.0, 20.0, 40.0, 60.0]
    area = compute_area(converted_bbox)
    assert area == 1200.0
    assert validate_box(converted_bbox) is True
    print("-> Geometry Utils: OK")

    # 3. Test matrix
    boxes_a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64)
    boxes_b = np.array([[0, 0, 10, 10], [0, 0, 20, 20]], dtype=np.float64)
    res = compute_all_matrices(boxes_a, boxes_b)
    assert res['iou'].shape == (2, 2)
    assert res['iou'][0, 0] == 1.0
    assert res['cont_a_in_b'][0, 1] == 1.0  # Box A0 (10x10) is inside Box B1 (20x20)
    print("-> Vectorized Matrix Utils: OK")

    # 4. Test schema loaders on sample b1_1020
    ocr_objs = load_ocr("b1_1020", BASE_DIR)
    small_objs = load_small("b1_1020", BASE_DIR)
    med_objs = load_medium("b1_1020", BASE_DIR)
    large_objs = load_large("b1_1020", BASE_DIR)
    cv_objs = load_cv("b1_1020", BASE_DIR)

    print(f"-> Schema Loader b1_1020: OCR={len(ocr_objs)}, Small={len(small_objs)}, Medium={len(med_objs)}, Large={len(large_objs)}, CV={len(cv_objs)}")
    assert len(ocr_objs) > 0
    assert len(small_objs) > 0
    assert len(cv_objs) > 0

    # Verify CV bbox conversion
    cv_sample = cv_objs[0]
    print(f"-> Sample CV object (converted to xyxy): {cv_sample}")
    assert cv_sample['source'] == 'cv'
    assert len(cv_sample['bbox']) == 4
    print("-> Schema Loaders: OK")

    print("=== ALL CORE TESTS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    test_core_module()
