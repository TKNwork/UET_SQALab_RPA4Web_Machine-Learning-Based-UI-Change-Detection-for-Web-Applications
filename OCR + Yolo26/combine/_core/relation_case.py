from enum import Enum


class RelationCase(Enum):
    """
    Taxonomy Enum phân loại mối quan hệ không gian dùng chung cho tất cả các module cặp nguồn.
    """
    DUPLICATE       = "duplicate"         # IoU >= 0.85 AND AreaRatio >= 0.85
    A_NESTED_IN_B    = "a_nested_in_b"     # Containment(A in B) >= 0.95 AND Area(A) < Area(B)
    B_NESTED_IN_A    = "b_nested_in_a"     # Containment(B in A) >= 0.95 AND Area(B) < Area(A)
    PARTIAL_OVERLAP  = "partial_overlap"   # 0.5 <= Containment < 0.95 (ở một trong 2 chiều)
    STANDALONE       = "standalone"        # Containment < 0.5 ở CẢ 2 chiều
