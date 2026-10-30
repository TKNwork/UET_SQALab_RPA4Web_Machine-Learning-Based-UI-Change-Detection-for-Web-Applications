"""Shared detection normalization, evidence helpers and one-to-one assignment.

No GT/file-of-answers access. Assignment requires an explicit task cost builder.
"""
import json
import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
try:
    from PIL import Image
    _PIL_OK = True
except ImportError:
    _PIL_OK = False
GENERIC_LABELS = {"", "ui_element", "uielement", "rectangle_box", "rectanglebox", "divider"}
OCR_SOURCES = {"ocr"}
NORMALIZATION_CONFIG = {
    "anchor_min_text_sim": .92, "anchor_min_text_len": 3, "anchor_min_confidence": .5,
    "dup_iou_threshold": .70, "dup_contain_threshold": .85,
    "container_min_shared_anchors": 1, "container_max_cost": 3.5,
}

def normalize_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_ocr_text(elem: dict) -> str:
    """
    Extract OCR text from an element based on its actual source field.
    Only treat class_label as OCR if source is in OCR_SOURCES.
    Never fabricate text.
    """
    src = elem.get("source", "")
    label = elem.get("class_label", "") or ""
    if src in OCR_SOURCES:
        norm = normalize_text(label)
        if norm and norm not in GENERIC_LABELS:
            return norm
    return ""


def extract_description(elem: dict) -> str:
    """
    Extract semantic description if present. In the current dataset
    no 'description' field exists, so this returns "".
    Never fabricate.
    """
    desc = elem.get("description") or elem.get("desc") or ""
    if not desc or not isinstance(desc, str):
        return ""
    norm = normalize_text(desc)
    if norm in GENERIC_LABELS or norm in {"medium", "ocr", "small", "large", "cv"}:
        return ""
    return norm


def normalize_detection(elem: dict, orig_id: str) -> dict:
    """
    Produce a normalized record from a raw element dict.
    Keeps a back-reference to the original detection ID.
    """
    bbox = elem.get("bbox", [])
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox for {orig_id}: {bbox}")

    x1, y1, x2, y2 = (float(v) for v in bbox)
    if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate bbox for {orig_id}: {bbox}")

    width  = x2 - x1
    height = y2 - y1

    ocr_text    = extract_ocr_text(elem)
    description = extract_description(elem)

    return {
        "orig_id":     orig_id,
        "id":          elem.get("id", orig_id),
        "source":      elem.get("source", ""),
        "class_label": elem.get("class_label", ""),
        "ocr_text":    ocr_text,
        "description": description,
        "has_ocr":     bool(ocr_text),
        "has_desc":    bool(description),
        "confidence":  float(elem["confidence"]) if elem.get("confidence") is not None else None,
        "class_name": elem.get("class_name", elem.get("class_label", "")) if not ocr_text else "text",
        "cluster_size": int(elem.get("cluster_size", 1)),
        "bbox":        [round(x1,2), round(y1,2), round(x2,2), round(y2,2)],
        "cx":          round((x1+x2)/2, 2),
        "cy":          round((y1+y2)/2, 2),
        "width":       round(width, 2),
        "height":      round(height, 2),
        "area":        round(width*height, 2),
        "aspect_ratio": round(width/height, 6) if height>0 else 0.0,
    }


def load_normalized(json_path: Path) -> tuple[list[dict], list[str]]:
    """Load and normalize all valid detections from a JSON file."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Cannot load {json_path}: {e}") from e

    elems_raw = data.get("elements", [])
    normed = []
    warnings = []
    for idx, elem in enumerate(elems_raw):
        eid = elem.get("id", f"__idx_{idx}")
        try:
            normed.append(normalize_detection(elem, eid))
        except ValueError as exc:
            warnings.append(str(exc))

    if len({e["id"] for e in normed}) != len(normed):
        raise ValueError(f"Duplicate detection IDs in {json_path}")
    return normed, warnings


# ===========================================================================
# GEOMETRY HELPERS
# ===========================================================================

def bbox_iou(a: list, b: list) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw  = max(0.0, ix2-ix1); ih = max(0.0, iy2-iy1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def containment(inner: list, outer: list) -> float:
    """Fraction of inner bbox area that is inside outer bbox."""
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    iw  = max(0.0, ix2-ix1); ih = max(0.0, iy2-iy1)
    inter = iw*ih
    inner_area = (inner[2]-inner[0])*(inner[3]-inner[1])
    return inter/inner_area if inner_area > 0 else 0.0


def center_dist(a: dict, b: dict) -> float:
    return math.hypot(b["cx"]-a["cx"], b["cy"]-a["cy"])


def size_ratio(a: dict, b: dict) -> float:
    wr = max(a["width"],b["width"]) / max(min(a["width"],b["width"]), 1e-6)
    hr = max(a["height"],b["height"]) / max(min(a["height"],b["height"]), 1e-6)
    return max(wr, hr)


def log_size_diff(a: dict, b: dict) -> float:
    lw = abs(math.log(max(a["width"],1e-6)/max(b["width"],1e-6)))
    lh = abs(math.log(max(a["height"],1e-6)/max(b["height"],1e-6)))
    return 0.5*(lw+lh)


def edge_shift(bb: list, ab: list) -> float:
    return max(abs(ab[k]-bb[k]) for k in range(4))


# ===========================================================================
# B. NEAR-DUPLICATE AND PARENT-CHILD GROUPING
# ===========================================================================

def build_duplicate_groups(elems: list[dict], cfg: dict) -> dict:
    """
    Group elements that are near-duplicates of each other vs. parent-child.

    Returns:
        {
          "groups": [{"group_id":..., "members":[id,...], "reason":...},...],
          "id_to_group": {elem_id: group_id},
          "parent_child": [{parent:id, child:id, containment:float},...],
          "representative": {group_id: elem_id}  # largest by area
        }
    """
    n = len(elems)
    dup_threshold  = cfg["dup_iou_threshold"]
    cont_threshold = cfg["dup_contain_threshold"]

    # Union-Find for duplicates
    parent = list(range(n))
    reason_map: dict[tuple, str] = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y, reason):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
            reason_map[(min(x,y), max(x,y))] = reason

    parent_child_pairs = []

    for i in range(n):
        for j in range(i+1, n):
            a, b = elems[i], elems[j]
            iou = bbox_iou(a["bbox"], b["bbox"])
            ca  = containment(a["bbox"], b["bbox"])  # a inside b
            cb  = containment(b["bbox"], a["bbox"])  # b inside a

            # Parent-child: one clearly inside the other, but not near-duplicate
            if (ca >= cont_threshold or cb >= cont_threshold) and iou < dup_threshold:
                inner, outer = (i,j) if ca >= cb else (j,i)
                parent_child_pairs.append({
                    "parent": elems[outer]["id"],
                    "child":  elems[inner]["id"],
                    "parent_idx": outer,
                    "child_idx":  inner,
                    "containment": round(max(ca,cb), 4)
                })
                continue  # NOT merged as duplicate

            # Near-duplicate: same element detected multiple times
            if iou >= dup_threshold and a["has_ocr"] == b["has_ocr"]:
                # Same text? additional confirmation
                oa, ob = a["ocr_text"], b["ocr_text"]
                if oa and ob:
                    sim = SequenceMatcher(None, oa, ob).ratio()
                    if sim < cfg["anchor_min_text_sim"]:
                        continue  # different text → not dup
                reason = f"iou={iou:.3f}"
                if oa and ob:
                    reason += f" text_sim={sim:.3f}"
                union(i, j, reason)

    # Build groups
    groups_map: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        groups_map[find(idx)].append(idx)

    groups = []
    id_to_group = {}
    representative = {}

    for gid_idx, (root, members) in enumerate(groups_map.items()):
        gid = f"dup_group_{gid_idx}"
        member_ids = [elems[m]["id"] for m in members]
        # Representative = largest area
        rep_idx = max(members, key=lambda m: elems[m]["area"])
        rep_id  = elems[rep_idx]["id"]

        member_set = set(members)
        reason = "; ".join(reason for (i,j),reason in sorted(reason_map.items()) if i in member_set and j in member_set) or "singleton"

        groups.append({
            "group_id":     gid,
            "members":      member_ids,
            "representative": rep_id,
            "reason":       reason,
            "original_bboxes": {elems[m]["id"]: elems[m]["bbox"] for m in members},
            "is_singleton": len(members) == 1
        })
        for mid in member_ids:
            id_to_group[mid] = gid
        representative[gid] = rep_id

    return {
        "groups":         groups,
        "id_to_group":    id_to_group,
        "representative": representative,
        "parent_child":   parent_child_pairs,
    }


def group_representative_elems(elems: list[dict], dup_info: dict) -> list[dict]:
    """
    Return one element per duplicate group (the representative).
    Keeps a 'group_members' field listing all member IDs.
    """
    rep_ids = set(dup_info["representative"].values())
    id_to_elem = {e["id"]: e for e in elems}
    result = []
    for grp in dup_info["groups"]:
        rep_id = grp["representative"]
        if rep_id not in id_to_elem:
            continue
        elem = dict(id_to_elem[rep_id])
        elem["group_id"]      = grp["group_id"]
        elem["group_members"] = grp["members"]
        elem["ocr_members"] = [{"id": mid, "text": id_to_elem[mid]["ocr_text"]} for mid in grp["members"] if id_to_elem[mid]["has_ocr"]]
        elem["original_bboxes"] = {mid: id_to_elem[mid]["bbox"] for mid in grp["members"]}
        result.append(elem)
    return result


# ===========================================================================
# C. OCR ANCHORS
# ===========================================================================

def build_ocr_anchors(b_elems: list[dict], a_elems: list[dict],
                      cfg: dict) -> dict:
    """
    Find confident OCR-based correspondences.

    Rules:
    - Text must be non-empty and >= min_text_len after normalize.
    - Prioritize exact matches that are unique in both images.
    - Near-exact (sim >= anchor_min_text_sim) kept as candidates.
    - Repeated text in either image: mark as ambiguous, do not lock.
    - Geometry sanity: check but do not enforce IoU (elements may move).

    Returns:
        {
          "locked":    [{before_id, after_id, ocr_before, ocr_after,
                         sim, geometry_ok, evidence},...],
          "candidates":[same schema but ambiguous — kept for scoring],
          "skipped":   [{before_id, reason},...],
          "anchor_shift": {dx, dy} or None if < 2 locked anchors
        }
    """
    min_len = cfg["anchor_min_text_len"]
    min_sim = cfg["anchor_min_text_sim"]

    b_ocr = [(e, e["ocr_text"]) for e in b_elems if e["has_ocr"]
              and len(e["ocr_text"]) >= min_len]
    a_ocr = [(e, e["ocr_text"]) for e in a_elems if e["has_ocr"]
              and len(e["ocr_text"]) >= min_len]

    # Count text occurrences in each side
    b_count: dict[str, int] = defaultdict(int)
    a_count: dict[str, int] = defaultdict(int)
    for _, t in b_ocr:
        b_count[t] += 1
    for _, t in a_ocr:
        a_count[t] += 1

    locked    = []
    candidates= []
    skipped   = []
    used_b    = set()
    used_a    = set()

    # Build sim matrix
    for b_elem, b_text in b_ocr:
        best_sim  = -1.0
        best_match= None

        for a_elem, a_text in a_ocr:
            if b_text == a_text:
                sim = 1.0
            else:
                sim = SequenceMatcher(None, b_text, a_text).ratio()
            if sim >= min_sim and sim > best_sim:
                best_sim  = sim
                best_match= (a_elem, a_text, sim)

        if best_match is None:
            skipped.append({"before_id": b_elem["id"], "ocr": b_text,
                             "reason": "no_after_match"})
            continue

        a_elem, a_text, sim = best_match
        b_unique = (b_count[b_text] == 1)
        a_unique = (a_count[a_text] == 1)

        # Geometry sanity (not enforced as gate, just recorded)
        dist  = center_dist(b_elem, a_elem)
        sr    = size_ratio(b_elem, a_elem)
        geo_ok = sr <= 6.0  # very loose — element may move significantly

        record = {
            "before_id":   b_elem["id"],
            "after_id":    a_elem["id"],
            "ocr_before":  b_text,
            "ocr_after":   a_text,
            "sim":         round(sim, 4),
            "b_unique":    b_unique,
            "a_unique":    a_unique,
            "geometry_ok": geo_ok,
            "center_dist": round(dist, 2),
            "size_ratio":  round(sr, 4),
            "evidence":    f"ocr_sim={sim:.3f}, unique_b={b_unique}, unique_a={a_unique}",
        }

        if b_unique and a_unique and geo_ok and sim == 1.0 and all(e["confidence"] is None or e["confidence"] >= cfg["anchor_min_confidence"] for e in (b_elem,a_elem)):
            if b_elem["id"] not in used_b and a_elem["id"] not in used_a:
                locked.append(record)
                used_b.add(b_elem["id"])
                used_a.add(a_elem["id"])
            else:
                record["reason"] = "id_already_locked"
                candidates.append(record)
        else:
            record["reason"] = ("repeated_text" if not (b_unique and a_unique)
                                else "near_exact_or_geometry_sanity")
            candidates.append(record)

    # Estimate anchor shift from locked pairs
    anchor_shift = None
    if len(locked) >= 2:
        id_to_b = {e["id"]: e for e in b_elems}
        id_to_a = {e["id"]: e for e in a_elems}
        dxs, dys = [], []
        for anc in locked:
            be = id_to_b.get(anc["before_id"])
            ae = id_to_a.get(anc["after_id"])
            if be and ae:
                dxs.append(ae["cx"] - be["cx"])
                dys.append(ae["cy"] - be["cy"])
        if dxs:
            anchor_shift = {"dx": round(float(np.median(dxs)), 2),
                            "dy": round(float(np.median(dys)), 2),
                            "n_anchors": len(dxs)}

    return {
        "locked":       locked,
        "candidates":   candidates,
        "skipped":      skipped,
        "anchor_shift": anchor_shift,
        "local_samples": [{"x": e["cx"], "y": e["cy"], "dx": f["cx"]-e["cx"], "dy": f["cy"]-e["cy"]} for anc in locked for e in b_elems for f in a_elems if e["id"]==anc["before_id"] and f["id"]==anc["after_id"]],
    }


# ===========================================================================
# D. CONTAINER MATCHING VIA CHILD CONTENT
# ===========================================================================

def get_contained_ids(container: dict, elems: list[dict],
                      threshold: float = 0.80) -> set[str]:
    """Return IDs of elements that are mostly inside this container."""
    result = set()
    for e in elems:
        if e["id"] == container["id"]:
            continue
        if containment(e["bbox"], container["bbox"]) >= threshold:
            result.add(e["id"])
    return result


def match_containers(b_elems, a_elems, anchors, cfg):
    """Return ALL feasible container candidates; assignment resolves competition."""
    records = []
    for be in [e for e in b_elems if not e["has_ocr"]]:
        bc = get_contained_ids(be, b_elems)
        expected = [anc for anc in anchors["locked"] if anc["before_id"] in bc]
        if len(expected) < cfg["container_min_shared_anchors"]:
            continue
        for ae in [e for e in a_elems if not e["has_ocr"]]:
            ac = get_contained_ids(ae, a_elems)
            shared = [anc for anc in expected if anc["after_id"] in ac]
            if len(shared) < cfg["container_min_shared_anchors"]:
                continue
            bl = {e["id"]: e for e in b_elems}; al = {e["id"]: e for e in a_elems}
            layout = []
            for anc in shared:
                x, y = bl[anc["before_id"]], al[anc["after_id"]]
                layout.append(math.hypot((x["cx"]-be["bbox"][0])/be["width"]-(y["cx"]-ae["bbox"][0])/ae["width"], (x["cy"]-be["bbox"][1])/be["height"]-(y["cy"]-ae["bbox"][1])/ae["height"]))
            score = len(shared)/len(expected)
            ratio = min(len(bc),len(ac))/max(len(bc),len(ac),1)
            cost = 2*(1-score)+(1-ratio)+.5*log_size_diff(be,ae)+float(np.mean(layout))
            if cost <= cfg["container_max_cost"]:
                records.append({"before_id":be["id"],"after_id":ae["id"],"cost":cost,"evidence":{"anchor_score":score,"child_ratio":ratio,"relative_child_layout_error":float(np.mean(layout)),"shared_ocrs":sorted(anc["ocr_before"] for anc in shared)}})
    return records


# ===========================================================================
# E. REMAINING ELEMENT MATCHING — MULTI-EVIDENCE
# ===========================================================================

def precompute_crop_histograms(elems: list[dict], img,
                               bins: int) -> dict[str, Optional[np.ndarray]]:
    """
    Precompute normalised per-channel histogram for each element's crop.
    Returns {elem_id: histogram_array (shape 3*bins,)} or None on failure.
    """
    result: dict[str, Optional[np.ndarray]] = {}
    if not _PIL_OK or img is None:
        for e in elems:
            result[e["id"]] = None
        return result
    try:
        w, h = img.size
        arr = np.array(img)
    except Exception:
        for e in elems:
            result[e["id"]] = None
        return result

    for e in elems:
        try:
            x1,y1,x2,y2 = [int(v) for v in e["bbox"]]
            x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
            if x2<=x1 or y2<=y1:
                result[e["id"]] = None
                continue
            crop_arr = arr[y1:y2, x1:x2]
            if crop_arr.size == 0:
                result[e["id"]] = None
                continue
            # Resize to 32x32 for speed
            from PIL import Image as _Img
            crop_img = _Img.fromarray(crop_arr).resize((32,32))
            crop_arr2 = np.array(crop_img)
            hists = []
            for ch in range(3):
                h_arr = np.histogram(crop_arr2[:,:,ch],
                                     bins=bins, range=(0,256))[0].astype(float)
                h_arr /= (h_arr.sum()+1e-9)
                hists.append(h_arr)
            result[e["id"]] = np.concatenate(hists)
        except Exception:
            result[e["id"]] = None
    return result


def crop_histogram_sim_precomputed(b_hist: Optional[np.ndarray],
                                   a_hist: Optional[np.ndarray]) -> Optional[float]:
    """Histogram intersection similarity from precomputed histograms."""
    if b_hist is None or a_hist is None:
        return None
    sim = float(np.sum(np.minimum(b_hist, a_hist)))
    # Each of 3 channels sums to 1 → max sim = 3, normalise to [0,1]
    return round(sim / 3.0, 4)


def solve_assignment(before_reps: list[dict], after_reps: list[dict],
                     pre_matched_b: set[str], pre_matched_a: set[str],
                     diagonal: float,
                     anchors: dict,
                     cfg: dict,
                     b_hists: Optional[dict] = None,
                     a_hists: Optional[dict] = None,
                     cost_builder=None) -> dict:
    """
    Solve one-to-one assignment for elements not yet pre-matched.
    Uses dummy nodes (unmatched penalty) so elements can remain unmatched.

    Returns:
        {
          "matched_pairs":    [match record,...],
          "uncertain_pairs":  [match record,...],
          "unmatched_before": [elem,...],
          "unmatched_after":  [elem,...],
          "cost_matrix_info": {...},   # for diagnostics
        }
    """
    if cost_builder is None:
        raise ValueError("Assignment requires a task-specific cost_builder")
    b_pool = [e for e in before_reps if e["id"] not in pre_matched_b]
    a_pool = [e for e in after_reps  if e["id"] not in pre_matched_a]

    if not b_pool or not a_pool:
        return {
            "matched_pairs":   [],
            "uncertain_pairs": [],
            "unmatched_before": b_pool,
            "unmatched_after":  a_pool,
            "cost_matrix_info": {},
        }

    n_b = len(b_pool)
    n_a = len(a_pool)

    # Build cost matrix with dummy (n_b + n_a) square extension
    dummy_cost = cfg["unmatched_cost"]
    max_cost   = cfg["max_match_cost"]
    sz = n_b + n_a
    C = np.full((sz, sz), 1e9, dtype=np.float64)
    C[n_b:, n_a:] = 0.0
    for i in range(n_b): C[i, n_a+i] = dummy_cost / 2
    for j in range(n_a): C[n_b+j, j] = dummy_cost / 2
    trace = []

    # Fill real pairs
    raw_costs: dict[tuple[int,int], float] = {}
    for ri, be in enumerate(b_pool):
        for ci, ae in enumerate(a_pool):
            detail = cost_builder(be, ae, diagonal, anchors, cfg,
                                       b_hists, a_hists, return_detail=True)
            cost = detail["cost"] if detail else None
            trace.append({"before_id":be["id"],"after_id":ae["id"],"feasible":cost is not None and cost <= min(max_cost,dummy_cost),"evidence":detail})
            if cost is not None and cost <= min(max_cost, dummy_cost):
                C[ri, ci] = cost
                raw_costs[(ri, ci)] = cost

    # Each unmatched side pays half the rejection cost; dummy-dummy costs zero.
    row_ind, col_ind = linear_sum_assignment(C)
    optimum = float(C[row_ind, col_ind].sum())

    matched    = []
    uncertain  = []
    used_b_idx = set()
    used_a_idx = set()

    for ri, ci in zip(row_ind, col_ind):
        if ri >= n_b or ci >= n_a:
            continue  # dummy assignment
        c_val = C[ri, ci]
        if c_val >= dummy_cost:
            continue  # effectively unmatched

        be = b_pool[ri]
        ae = a_pool[ci]

        # ── Ambiguity check ────────────────────────────────────────────────
        # Row: other valid assignments for this before element
        r_others = [C[ri, c] for c in range(n_a) if c != ci and C[ri,c] < dummy_cost]
        # Col: other valid assignments for this after element
        c_others = [C[r, ci] for r in range(n_b) if r != ri and C[r,ci] < dummy_cost]

        second_r = min(r_others) if r_others else 1e9
        second_c = min(c_others) if c_others else 1e9
        margin   = min(second_r - c_val if second_r < 1e8 else 999.0,
                       second_c - c_val if second_c < 1e8 else 999.0)

        # ── Stability check: ban best edge and re-examine ──────────────────
        C_ban = C.copy()
        C_ban[ri, ci] = 1e9
        r_ban, c_ban  = linear_sum_assignment(C_ban)
        cost_gap = float(C_ban[r_ban,c_ban].sum() - optimum)

        evidence = {
            "cost":             round(float(c_val), 4),
            "margin":           round(float(margin), 4) if margin < 900 else None,
            "cost_gap_ban":     round(cost_gap, 4),
            "iou":              round(bbox_iou(be["bbox"], ae["bbox"]), 4),
            "center_dist_px":   round(center_dist(be, ae), 2),
            "log_size_diff":    round(log_size_diff(be, ae), 4),
            "b_ocr":            be.get("ocr_text",""),
            "a_ocr":            ae.get("ocr_text",""),
            "b_group_members":  be.get("group_members",[]),
            "a_group_members":  ae.get("group_members",[]),
            **next((t["evidence"] for t in trace if t["before_id"]==be["id"] and t["after_id"]==ae["id"]),{}),
        }

        record = {
            "before_id":         be["id"],
            "after_id":          ae["id"],
            "before_bbox":       be["bbox"],
            "after_bbox":        ae["bbox"],
            "before_group_id":   be.get("group_id",""),
            "after_group_id":    ae.get("group_id",""),
            "match_type":        "evidence_assignment",
            "evidence":          evidence,
        }

        if cost_gap >= cfg["stability_ban_cost_gap"]:
            matched.append(record)
        else:
            record["uncertain_reason"] = (
                f"global stability gap below {cfg['stability_ban_cost_gap']}; margin={margin:.4f}; "
                f"cost_gap={cost_gap:.4f}"
            )
            uncertain.append(record)

        used_b_idx.add(ri)
        used_a_idx.add(ci)

    unmatched_b = [b_pool[i] for i in range(n_b) if i not in used_b_idx]
    unmatched_a = [a_pool[i] for i in range(n_a) if i not in used_a_idx]

    return {
        "matched_pairs":    matched,
        "uncertain_pairs":  uncertain,
        "unmatched_before": unmatched_b,
        "unmatched_after":  unmatched_a,
        "cost_matrix_info": {
            "n_before": n_b, "n_after": n_a,
            "n_feasible_edges": len(raw_costs),
            "anchor_shift": anchors.get("anchor_shift"),
            "candidate_trace": trace,
            "optimum_assignment_cost": optimum,
        },
    }


# ===========================================================================
# DELTA CALCULATION (same-level, geometry only)
# ===========================================================================

def calc_delta(be: dict, ae: dict, diagonal: float) -> dict:
    dx = ae["cx"] - be["cx"]
    dy = ae["cy"] - be["cy"]
    dw = ae["width"]  - be["width"]
    dh = ae["height"] - be["height"]
    rel_dw = dw / be["width"]  if be["width"]  > 0 else 0.0
    rel_dh = dh / be["height"] if be["height"] > 0 else 0.0
    return {
        "dx":          round(dx, 2),
        "dy":          round(dy, 2),
        "dw":          round(dw, 2),
        "dh":          round(dh, 2),
        "relative_dw": round(rel_dw, 4),
        "relative_dh": round(rel_dh, 4),
        "relative_dx": round(dx/diagonal, 4) if diagonal > 0 else 0.0,
        "relative_dy": round(dy/diagonal, 4) if diagonal > 0 else 0.0,
        "center_shift": round(math.hypot(dx,dy), 2),
        "max_edge_shift": round(edge_shift(be["bbox"], ae["bbox"]), 2),
    }


# ===========================================================================
# G. CLASSIFIER (three modes, decoupled from matching)
# ===========================================================================

