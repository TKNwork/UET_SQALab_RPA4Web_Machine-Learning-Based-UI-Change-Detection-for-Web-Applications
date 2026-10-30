# combine/_core/thresholds.py

# Ngưỡng mặc định cho Taxonomy RelationCase
IOU_DUPLICATE_THRESH: float = 0.85
AREA_RATIO_DUPLICATE_THRESH: float = 0.85

CONTAINMENT_NESTED_THRESH: float = 0.95
CONTAINMENT_PARTIAL_THRESH: float = 0.50

# Ngưỡng lọc theo kích thước COCO (diện tích px^2)
SIZE_SMALL_MAX_AREA: float = 1024.0      # Area < 1024
SIZE_MEDIUM_MAX_AREA: float = 9216.0     # 1024 <= Area < 9216
