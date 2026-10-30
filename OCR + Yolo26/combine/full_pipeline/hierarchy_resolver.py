import logging
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger("full_pipeline.hierarchy_resolver")


def resolve_hierarchy(
    all_nodes: Dict[str, Dict[str, Any]],
    all_edges: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Dựng cây phân cấp UI Element từ đồ thị các node và edges:
    1. Multi-parent detection
    2. Select primary_parent (Tightest Containment - diện tích nhỏ nhất trong các box chứa)
    3. Ancestor chain traversal
    4. hierarchy_gap flag detection (nếu Small/OCR/CV trực tiếp thuộc Large mà không qua Medium)
    5. Giữ trọn vẹn secondary_relations (duplicate, partial_overlap, v.v.)
    """
    # 1. Thu thập danh sách các box chứa (container candidates) cho từng node
    containers_map: Dict[str, List[Tuple[str, float, str]]] = {node_id: [] for node_id in all_nodes}
    secondary_map: Dict[str, List[Dict[str, Any]]] = {node_id: [] for node_id in all_nodes}

    for edge in all_edges:
        a_id = edge.get("a_id")
        b_id = edge.get("b_id")
        case = edge.get("case")

        if a_id not in all_nodes or b_id not in all_nodes:
            continue

        node_a = all_nodes[a_id]
        node_b = all_nodes[b_id]

        if case == "a_nested_in_b":
            # A nằm trong B -> B là container candidate của A
            containers_map[a_id].append((b_id, node_b['area'], case))
        elif case == "b_nested_in_a":
            # B nằm trong A -> A là container candidate của B
            containers_map[b_id].append((a_id, node_a['area'], case))
        else:
            # duplicate hoặc partial_overlap -> lưu vào secondary_relations cho cả 2
            secondary_map[a_id].append({
                "id": b_id,
                "source": node_b['source'],
                "case": case,
                "metrics": edge.get("metrics")
            })
            secondary_map[b_id].append({
                "id": a_id,
                "source": node_a['source'],
                "case": case,
                "metrics": edge.get("metrics")
            })

    # 2. Chọn primary_parent cho từng node (Tightest Containment - Area nhỏ nhất)
    primary_parent_map: Dict[str, Optional[Dict[str, Any]]] = {}

    for node_id, cand_list in containers_map.items():
        if not cand_list:
            primary_parent_map[node_id] = None
        else:
            # Sắp xếp các container theo diện tích tăng dần
            cand_list.sort(key=lambda x: x[1])
            best_parent_id, best_area, rel_case = cand_list[0]
            parent_node = all_nodes[best_parent_id]
            primary_parent_map[node_id] = {
                "id": best_parent_id,
                "source": parent_node["source"],
                "case": rel_case
            }

    # 3. Chuỗi tổ tiên (Ancestors chain) & Cờ hierarchy_gap
    resolved_elements = []

    for node_id, node_data in all_nodes.items():
        # Truy vết chuỗi parent ngược lên root
        ancestors = []
        curr_id = node_id
        visited = set()

        while curr_id in primary_parent_map and primary_parent_map[curr_id] is not None:
            parent_info = primary_parent_map[curr_id]
            p_id = parent_info["id"]

            if p_id in visited:  # Chống vòng lặp
                break

            visited.add(p_id)
            ancestors.append(p_id)
            curr_id = p_id

        # 4. Kiểm tra hierarchy_gap flag
        # Rào chắn: Nếu node là 'small', 'ocr', hoặc 'cv', có ancestor chứa 'large' nhưng KHÔNG có 'medium' trong chuỗi ancestors
        hierarchy_gap = False
        ancestor_sources = [all_nodes[aid]["source"] for aid in ancestors if aid in all_nodes]

        if node_data["source"] in ("small", "ocr", "cv"):
            if "large" in ancestor_sources and "medium" not in ancestor_sources:
                hierarchy_gap = True

        # 5. Đóng gói kết quả chuẩn hóa
        resolved_elements.append({
            "id": node_id,
            "source": node_data["source"],
            "class_label": node_data["label"],
            "bbox": node_data["bbox"],
            "confidence": node_data["confidence"],
            "area": node_data["area"],
            "text": node_data.get("text"),
            "primary_parent": primary_parent_map[node_id],
            "ancestors": ancestors,
            "secondary_relations": secondary_map[node_id],
            "hierarchy_gap": hierarchy_gap
        })

    return resolved_elements


def resolve_fused_hierarchy(fused_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dựng cây phân cấp UI Element hoàn chỉnh trên danh sách Fused Elements (đã qua WBF deduplication):
    1. Tính ma trận chứa (Tightest Containment - diện tích nhỏ nhất trong các box chứa)
    2. Gán primary_parent
    3. Truy vết ancestors
    4. Bổ sung danh sách children
    5. Xây dựng secondary_relations
    6. Đánh giá hierarchy_gap flag
    """
    if not fused_elements:
        return []

    import numpy as np
    from combine._core.matrix import compute_all_matrices
    from combine._core.thresholds import CONTAINMENT_NESTED_THRESH, CONTAINMENT_PARTIAL_THRESH

    num_elems = len(fused_elements)
    boxes = np.array([e['bbox'] for e in fused_elements], dtype=np.float64)

    matrix_dict = compute_all_matrices(boxes, boxes)
    iou_mat = matrix_dict['iou']
    cont_a_in_b_mat = matrix_dict['cont_a_in_b']

    containers_map: Dict[str, List[Tuple[str, float]]] = {e['id']: [] for e in fused_elements}
    secondary_map: Dict[str, List[Dict[str, Any]]] = {e['id']: [] for e in fused_elements}

    elem_map = {e['id']: e for e in fused_elements}

    for i in range(num_elems):
        id_i = fused_elements[i]['id']
        area_i = fused_elements[i]['area']

        for j in range(num_elems):
            if i == j:
                continue

            id_j = fused_elements[j]['id']
            area_j = fused_elements[j]['area']
            cont_i_in_j = float(cont_a_in_b_mat[i, j])
            iou_ij = float(iou_mat[i, j])

            if cont_i_in_j >= CONTAINMENT_NESTED_THRESH and area_i < area_j:
                containers_map[id_i].append((id_j, area_j))
            elif cont_i_in_j >= CONTAINMENT_PARTIAL_THRESH or iou_ij >= 0.34:
                secondary_map[id_i].append({
                    "id": id_j,
                    "case": "partial_overlap",
                    "metrics": {
                        "iou": round(iou_ij, 3),
                        "cont_a_in_b": round(cont_i_in_j, 3)
                    }
                })

    # 1. Chọn primary_parent cho từng fused element
    primary_parent_map: Dict[str, Optional[Dict[str, Any]]] = {}
    for elem_id, cand_list in containers_map.items():
        if not cand_list:
            primary_parent_map[elem_id] = None
        else:
            cand_list.sort(key=lambda x: x[1])  # Tightest containment (small area first)
            best_parent_id, _ = cand_list[0]
            primary_parent_map[elem_id] = {
                "id": best_parent_id,
                "case": "a_nested_in_b"
            }

    # 2. Xây dựng ancestors chain
    ancestors_map: Dict[str, List[str]] = {}
    for elem in fused_elements:
        curr_id = elem['id']
        anc_list = []
        visited = set()

        while curr_id in primary_parent_map and primary_parent_map[curr_id] is not None:
            p_info = primary_parent_map[curr_id]
            p_id = p_info["id"]

            if p_id in visited:
                break

            visited.add(p_id)
            anc_list.append(p_id)
            curr_id = p_id

        ancestors_map[elem['id']] = anc_list

    # 3. Xây dựng mảng children
    children_map: Dict[str, List[str]] = {e['id']: [] for e in fused_elements}
    for elem_id, p_info in primary_parent_map.items():
        if p_info is not None:
            p_id = p_info["id"]
            if p_id in children_map:
                children_map[p_id].append(elem_id)

    # 4. Tạo danh sách resolved elements hoàn chỉnh
    resolved_elements = []
    for elem in fused_elements:
        elem_id = elem['id']
        anc_ids = ancestors_map[elem_id]
        anc_sources = [elem_map[aid]['source'] for aid in anc_ids if aid in elem_map]

        hierarchy_gap = False
        if elem['source'] in ("small", "ocr", "cv"):
            if "large" in anc_sources and "medium" not in anc_sources:
                hierarchy_gap = True

        resolved_elements.append({
            "id": elem_id,
            "source": elem['source'],
            "sources": elem.get('sources', [elem['source']]),
            "class_label": elem.get('class_label', elem.get('label', 'ui_element')),
            "bbox": elem['bbox'],
            "confidence": elem['confidence'],
            "cluster_size": elem.get('cluster_size', 1),
            "area": elem['area'],
            "text": elem.get('text'),
            "primary_parent": primary_parent_map[elem_id],
            "ancestors": anc_ids,
            "children": children_map[elem_id],
            "secondary_relations": secondary_map[elem_id],
            "hierarchy_gap": hierarchy_gap
        })

    # 5. Gán integer element_id (0..N-1) bảo toàn tính liên tục và khớp với OmniParser
    fused_id_to_num = {e["id"]: idx for idx, e in enumerate(resolved_elements)}
    for idx, elem in enumerate(resolved_elements):
        elem["element_id"] = idx
        if elem["primary_parent"] is not None:
            parent_fused_id = elem["primary_parent"]["id"]
            elem["primary_parent"]["element_id"] = fused_id_to_num.get(parent_fused_id)

    return resolved_elements


