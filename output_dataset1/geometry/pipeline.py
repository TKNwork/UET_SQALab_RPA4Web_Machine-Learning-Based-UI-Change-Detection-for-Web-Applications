#!/usr/bin/env python3
"""
new_pipeline.py — Evidence-based element correspondence pipeline.

Matches GUI elements, groups changed correspondences and evaluates GT coverage.
Stages:
  A. Normalize detections (schema, source-aware OCR extraction)
  B. Near-duplicate & parent-child grouping
  C. OCR anchors (unique exact); near-exact stays candidate
  D. Container matching via contained anchors + child-content comparison
  E. Remaining element matching via multi-evidence cost
  F. One-to-one assignment with ambiguity handling and uncertain set

CLI:
  python3 new_pipeline.py --input-root <dir> --output-root <dir> --pair-ids 000 004 012 016 021
"""

import argparse
import hashlib
import time
import json
import math
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

# ── optional image libraries (crop similarity) ─────────────────────────────
try:
    from PIL import Image
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ===========================================================================
# CONSTANTS
# ===========================================================================

GENERIC_LABELS = {"", "ui_element", "uielement", "rectangle_box",
                  "rectanglebox", "divider"}

# Source tags that indicate the detector embedded OCR text in class_label
OCR_SOURCES = {"ocr"}

# Explicit defaults. This five-sample subset is a development set, not an independent test.
DEFAULT_CONFIG = {
    # Anchor building
    "anchor_min_text_sim": 0.92,   # min SequenceMatcher ratio for near-exact
    "anchor_min_text_len": 3,
    "anchor_min_confidence": 0.5,      # min chars after normalize

    # Duplicate grouping
    "dup_iou_threshold": 0.70,     # IoU threshold for near-duplicate candidates
    "dup_contain_threshold": 0.85, # containment ratio to call a box "inside" another

    # Container matching
    "container_min_shared_anchors": 1,
    "container_max_cost": 3.5,

    # General matching
    "max_pos_dist_ratio": 0.65,    # max center-dist / diagonal
    "max_size_ratio": 7.0,         # max(w1/w2, w2/w1)
    "anchor_evidence_weight": 2.0,
    "iou_evidence_weight": 1.5,
    "crop_evidence_weight": 0.8,
    "shape_evidence_weight": 0.5,
    "shift_prior_weight": 0.5,

    # Assignment
    "ambiguity_threshold": 0.10,   # diagnostic local margin; acceptance uses global stability
    "stability_ban_cost_gap": 0.05, # min extra cost when best edge is banned to call it stable
    "unmatched_cost": 2.5,         # dummy-node cost (elements below this prefer unmatched)
    "max_match_cost": 5.0,         # hard ceiling

    # OCR conflict rejection
    "ocr_conflict_sim": 0.45,      # reject pair if both have OCR but sim < this

    # Crop similarity
    "crop_histogram_bins": 32,     # histogram bins per channel for crop sim
}


# ===========================================================================
# A. NORMALIZE DETECTIONS
# ===========================================================================

from shared import matching as common_matching
from shared.matching import *
from shared.matching import _PIL_OK
def build_evidence_cost(b_elem: dict, a_elem: dict,
                        diagonal: float,
                        anchors: dict,
                        cfg: dict,
                        b_hists: Optional[dict] = None,
                        a_hists: Optional[dict] = None, return_detail=False):
    """
    Compute a single evidence-weighted cost for a Before–After pair.
    b_hists/a_hists: precomputed histogram dicts {elem_id: ndarray|None}.
    Returns None if the pair fails hard feasibility gates.
    """
    # ── feasibility gates ──────────────────────────────────────────────────
    dist   = center_dist(b_elem, a_elem)
    pos_r  = dist / diagonal if diagonal > 0 else 0.0
    sr     = size_ratio(b_elem, a_elem)

    if b_elem["has_ocr"] != a_elem["has_ocr"]:
        return None  # text and non-text use separate correspondence levels
    if pos_r > cfg["max_pos_dist_ratio"]:
        return None
    if sr > cfg["max_size_ratio"]:
        return None

    # OCR conflict gate
    if b_elem["has_ocr"] and a_elem["has_ocr"]:
        sim = SequenceMatcher(None, b_elem["ocr_text"],
                              a_elem["ocr_text"]).ratio()
        if sim < cfg["ocr_conflict_sim"]:
            return None

    # ── evidence components ────────────────────────────────────────────────
    costs = {}

    # 1. OCR / description similarity
    if b_elem["has_ocr"] and a_elem["has_ocr"]:
        sim = SequenceMatcher(None, b_elem["ocr_text"],
                              a_elem["ocr_text"]).ratio()
        costs["ocr"] = cfg["anchor_evidence_weight"] * (1.0 - sim)
    elif b_elem["has_desc"] and a_elem["has_desc"]:
        sim = SequenceMatcher(None, b_elem["description"],
                              a_elem["description"]).ratio()
        costs["desc"] = cfg["anchor_evidence_weight"] * (1.0 - sim)

    # 2. IoU (one evidence, not a gate)
    iou = bbox_iou(b_elem["bbox"], a_elem["bbox"])
    costs["iou"] = cfg["iou_evidence_weight"] * (1.0 - iou)

    # 3. Position distance
    costs["pos"] = 1.0 * pos_r

    # 4. Shape/size
    sd = log_size_diff(b_elem, a_elem)
    costs["size"] = cfg["shape_evidence_weight"] * sd

    # 5. Anchor shift prior (local shift from nearby anchors)
    samples = anchors.get("local_samples", [])
    nearby = sorted(samples, key=lambda v: math.hypot(v["x"]-b_elem["cx"],v["y"]-b_elem["cy"]))[:5]
    nearby = [v for v in nearby if math.hypot(v["x"]-b_elem["cx"],v["y"]-b_elem["cy"]) <= diagonal*.2]
    anc_shift = {"dx":float(np.median([v["dx"] for v in nearby])), "dy":float(np.median([v["dy"] for v in nearby]))} if nearby else None
    if anc_shift is not None:
        expected_cx = b_elem["cx"] + anc_shift["dx"]
        expected_cy = b_elem["cy"] + anc_shift["dy"]
        shift_err = math.hypot(a_elem["cx"] - expected_cx,
                               a_elem["cy"] - expected_cy)
        shift_err_r = shift_err / diagonal if diagonal > 0 else 0.0
        costs["shift_prior"] = cfg["shift_prior_weight"] * shift_err_r

    # 6. Crop similarity via precomputed histograms
    if b_hists is not None and a_hists is not None:
        bh = b_hists.get(b_elem["id"])
        ah = a_hists.get(a_elem["id"])
        crop_sim = crop_histogram_sim_precomputed(bh, ah)
        if crop_sim is not None and cfg["crop_evidence_weight"] > 0:
            costs["crop"] = cfg["crop_evidence_weight"] * (1.0 - crop_sim)

    container = next((r for r in anchors.get("container_candidates",[]) if r["before_id"]==b_elem["id"] and r["after_id"]==a_elem["id"]),None)
    if container:
        costs["container"] = -cfg["anchor_evidence_weight"] * container["evidence"]["anchor_score"] / (1+container["evidence"]["relative_child_layout_error"])
    total = sum(costs.values())
    detail = {"cost":total,"components":costs,"local_anchor_count":len(nearby),"confidence_before":b_elem["confidence"],"confidence_after":a_elem["confidence"],"container_evidence":container["evidence"] if container else None}
    return detail if return_detail else total


# ===========================================================================
# F. ASSIGNMENT WITH AMBIGUITY HANDLING
# ===========================================================================


def solve_assignment(*args, cost_builder=None, **kwargs):
    """Geometry cost by default; caller can explicitly choose another task cost."""
    return common_matching.solve_assignment(*args, cost_builder=cost_builder or build_evidence_cost, **kwargs)

def classify_change(delta: dict, mode: str) -> bool:
    """
    Modes: relative15 (center >8 px or relative resize >15%),
    edges8 (relative15 plus bbox edge displacement >8 px).
    """
    center_shift = delta["center_shift"]
    max_edge     = delta["max_edge_shift"]
    rel_resize   = max(abs(delta["relative_dw"]), abs(delta["relative_dh"]))

    if mode == "relative15":
        return center_shift > 8.0 or rel_resize > 0.15

    if mode == "edges8":
        return center_shift > 8.0 or rel_resize > 0.15 or max_edge > 8.0

    raise ValueError(f"Unknown classifier mode: {mode}")


# ===========================================================================
# H. EVALUATOR (independent, reads GT only here)
# ===========================================================================

def bbox_iou_xywh(g: dict, b: list) -> tuple[float, float]:
    """Compute IoU and CCR between GT {x,y,w,h} and pred bbox [x1,y1,x2,y2]."""
    gbox = [g["x"], g["y"], g["x"]+g["w"], g["y"]+g["h"]]
    ix1 = max(gbox[0], b[0]); iy1 = max(gbox[1], b[1])
    ix2 = min(gbox[2], b[2]); iy2 = min(gbox[3], b[3])
    iw  = max(0.0, ix2-ix1); ih = max(0.0, iy2-iy1)
    inter = iw*ih
    ga = g["w"]*g["h"]
    ba = (b[2]-b[0])*(b[3]-b[1])
    iou = inter/(ga+ba-inter+1e-9)
    ccr = inter/(ga+1e-9)
    return iou, ccr


EVALUATION_PROTOCOL = "envelope_coverage_v1"


def prediction_envelope(before_bbox, after_bbox):
    """Smallest rectangle enclosing both endpoints, without consulting GT."""
    for box in (before_bbox,after_bbox):
        if len(box)!=4 or not all(math.isfinite(float(v)) for v in box) or box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError(f"Invalid prediction bbox: {box}")
    return [min(before_bbox[0],after_bbox[0]),min(before_bbox[1],after_bbox[1]),
            max(before_bbox[2],after_bbox[2]),max(before_bbox[3],after_bbox[3])]


def evaluate(all_match_records: list[dict], gt_path: Path,
             classifier: str, protocol: str = EVALUATION_PROTOCOL) -> dict:
    """Each geometry GT needs >=1 changed matched-pair envelope; no one-to-one.

    assignments is a best-evidence view per GT, NOT a bipartite assignment.
    prediction_gt_links retains ALL accepted edges and counts unique supporting
    predictions, so unverified never means predictions minus covered GT.
    """
    if protocol != EVALUATION_PROTOCOL: raise ValueError(f"Unknown protocol: {protocol}")
    raw=json.loads(gt_path.read_text(encoding='utf-8')).get("bboxes",[])
    indexed=[(i,g) for i,g in enumerate(raw) if "geometry" in g.get("type",[])]
    selected=[m for m in all_match_records if "delta" in m and classify_change(m["delta"],classifier)]
    predictions=[{"prediction_index":i,"before_id":m["before_id"],"after_id":m["after_id"],
                  "before_bbox":m["before_bbox"],"after_bbox":m["after_bbox"],
                  "prediction_region":prediction_envelope(m["before_bbox"],m["after_bbox"])} for i,m in enumerate(selected)]
    links=[];by_gt={i:[] for i in range(len(indexed))}
    for p in predictions:
        for gi,(original_index,g) in enumerate(indexed):
            iou,ccr=bbox_iou_xywh(g,p["prediction_region"])
            if iou + 1e-12 < .30 and ccr + 1e-12 < .50: continue
            side_overlap={side:dict(zip(("iou","ccr"),bbox_iou_xywh(g,p[side+"_bbox"]))) for side in ("before","after")}
            edge={"gt_index":gi,"original_gt_index":original_index,**p,"iou":iou,"ccr":ccr,
                  "overlap":max(iou,ccr),"endpoint_overlap":side_overlap,
                  "envelope_gap_only":all(v["ccr"]==0 for v in side_overlap.values())}
            links.append(edge);by_gt[gi].append(edge)
    assignments=[max(es,key=lambda e:(e["overlap"],e["iou"],-e["prediction_index"])) for es in by_gt.values() if es]
    supported={e["prediction_index"] for e in links}
    return {"protocol":protocol,"classifier":classifier,"region_definition":"bbox_envelope_before_after",
            "coverage_rule":"Each geometry GT has at least one IoU>=0.30 or CCR>=0.50 edge; predictions may support multiple GT",
            "gt_count":len(indexed),"covered":len(assignments),"predictions":len(predictions),
            "supporting_predictions":len(supported),"unverified":len(predictions)-len(supported),
            "pass":len(assignments)==len(indexed),"assignments":assignments,"prediction_gt_links":links,
            "prediction_regions":predictions,"missed_gt":[i for i,es in by_gt.items() if not es],
            "gt_coverage":[{"gt_index":i,"original_gt_index":original_index,"selector":g.get("selector"),
                            "covered":bool(by_gt[i]),"prediction_indices":[e["prediction_index"] for e in by_gt[i]]}
                           for i,(original_index,g) in enumerate(indexed)]}


def evaluate_grouped(all_match_records, gt_path, classifier, group_data, mode="union_support"):
    """Add GT-independent groups to frozen singleton coverage.

    union_support: exact union of member envelopes must pass original thresholds.
    structural_proxy: also admit well-spread, coherent groups with >=3 distinct
    members, bbox passing old thresholds and >=10% GT actual union support.
    Proxy links are explicitly separate evidence, not confirmed correspondences.
    """
    from geometry.prediction_groups import area,intersect,union_area
    if mode not in ("union_support","structural_proxy"):raise ValueError(mode)
    base=evaluate(all_match_records,gt_path,classifier)
    assert group_data.get("uses_groundtruth") is False
    if group_data["selected_predictions"]!=base["predictions"]:
        raise ValueError("Groups do not match the selected prediction count")
    for group in group_data['groups']:
        for member in group['members']:
            index=member['prediction_index']
            if index<0 or index>=len(base['prediction_regions']):raise ValueError("Invalid group member index")
            original=base['prediction_regions'][index]
            if any(member[key]!=original[key] for key in ('before_id','after_id','before_bbox','after_bbox')):
                raise ValueError("Group member differs from frozen prediction")
    gt=[g for g in json.loads(Path(gt_path).read_text(encoding='utf-8')).get("bboxes",[]) if "geometry" in g.get("type",[])]
    cfg=group_data["config"];links=[]
    for group in group_data["groups"]:
        for i,g in enumerate(gt):
            box=[g['x'],g['y'],g['x']+g['w'],g['y']+g['h']]
            inter=union_area([intersect(b,box) for b in group['member_regions']])
            support_ccr=inter/max(area(box),1e-9)
            support_iou=inter/max(area(box)+group['support_area']-inter,1e-9)
            bbox_iou,bbox_ccr=bbox_iou_xywh(g,group['prediction_region'])
            exact=support_iou+1e-12>=.3 or support_ccr+1e-12>=.5
            proxy=(mode=="structural_proxy" and not exact and len(group['members'])>=cfg['min_structural_members']
                   and (group['parent'] is not None or group['all_motion_compatible'])
                   and group['spatial_spread']>=cfg['structural_min_spread']
                   and support_ccr>=cfg['structural_min_support_ccr']
                   and (bbox_iou+1e-12>=.3 or bbox_ccr+1e-12>=.5))
            if exact or proxy:
                links.append({'gt_index':i,'group_id':group['group_id'],'kind':group['kind'],
                              'member_prediction_indices':group['member_prediction_indices'],
                              'supporting_member_prediction_indices':[member['prediction_index'] for member in group['members'] if area(intersect(prediction_envelope(member['before_bbox'],member['after_bbox']),box))>0],
                              'support_iou':support_iou,'support_ccr':support_ccr,'bbox_iou':bbox_iou,'bbox_ccr':bbox_ccr,
                              'evidence_type':'union_support' if exact else 'structural_proxy','needs_human_review':True})
    covered_single={g['gt_index'] for g in base['gt_coverage'] if g['covered']}
    covered=covered_single|{l['gt_index'] for l in links}
    newly=sorted(covered-covered_single)
    result=dict(base,protocol="group_coverage_v1_"+mode,base_protocol=base['protocol'],
                single_covered=len(covered_single),covered=len(covered),pass_=len(covered)==len(gt),
                group_count=len(group_data['groups']),group_links=links,group_rescued_gt=newly,
                missed_gt=sorted(set(range(len(gt)))-covered))
    result['pass']=result.pop('pass_')
    supported={l['prediction_index'] for l in base['prediction_gt_links']}
    for link in links:supported.update(link['supporting_member_prediction_indices'])
    result['supporting_predictions']=len(supported);result['unverified']=base['predictions']-len(supported)
    single_assignments={a['gt_index']:a for a in base['assignments']}
    for gi in newly:
        best=max((l for l in links if l['gt_index']==gi),key=lambda l:(l['evidence_type']=='union_support',l['support_iou'],l['support_ccr']))
        single_assignments[gi]={'gt_index':gi,'evidence_type':best['evidence_type'],**best}
    result['assignments']=[single_assignments[gi] for gi in sorted(single_assignments)]
    result['gt_coverage']=[dict(row,covered=row['gt_index'] in covered,
                              single_prediction_indices=row['prediction_indices'],
                              prediction_indices=sorted(set(row['prediction_indices'])|{p for l in links if l['gt_index']==row['gt_index'] for p in l['supporting_member_prediction_indices']}),
                              group_ids=[l['group_id'] for l in links if l['gt_index']==row['gt_index']]) for row in base['gt_coverage']]
    result['coverage_rule']='singleton envelope OR group '+mode+'; no one-to-one GT assignment'
    return result


# ===========================================================================
# MAIN PROCESSING FUNCTION
# ===========================================================================

def resolve_images(pair_dir: Path, image_root=None):
    local = {"before":pair_dir/"before.png", "after":pair_dir/"after.png"}
    if all(p.is_file() for p in local.values()):
        return local
    if image_root:
        matches = [p for p in Path(image_root).glob(pair_dir.name+"_*") if p.is_dir() and (p/"baseline.png").is_file() and (p/"inject.png").is_file()]
        if len(matches) != 1:
            raise ValueError(f"Expected unique image source for {pair_dir.name}, found {len(matches)}")
        return {"before":matches[0]/"baseline.png", "after":matches[0]/"inject.png"}
    return local


def process_pair(pair_dir: Path, output_dir: Path, cfg: dict,
                 ablation: str = "full") -> dict:
    """
    Run the new pipeline on one Before–After pair.

    ablation:
      'full'              — complete new pipeline
      'no_duplicate'      — skip duplicate grouping (treat each det as-is)
      'no_anchor'         — skip OCR anchor + container evidence
      'no_crop'           — skip crop similarity
    """
    pair_id = pair_dir.name
    out_dir = output_dir / pair_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    b_elems_raw, b_warn = load_normalized(pair_dir / "before.json")
    a_elems_raw, a_warn = load_normalized(pair_dir / "after.json")

    image_paths = resolve_images(pair_dir, cfg.get("image_root"))
    # Load images and precompute crop histograms
    b_img = a_img = None
    bins = cfg["crop_histogram_bins"]
    if _PIL_OK:
        bp = image_paths["before"]
        if bp.exists():
            try:
                b_img = Image.open(bp).convert("RGB")
            except Exception:
                pass
        ap = image_paths["after"]
        if ap.exists():
            try:
                a_img = Image.open(ap).convert("RGB")
            except Exception:
                pass
    b_hists = precompute_crop_histograms(b_elems_raw, b_img, bins)
    a_hists = precompute_crop_histograms(a_elems_raw, a_img, bins)

    # ── Diagonal ──────────────────────────────────────────────────────────
    all_x2 = [e["bbox"][2] for e in b_elems_raw + a_elems_raw]
    all_y2 = [e["bbox"][3] for e in b_elems_raw + a_elems_raw]
    max_x  = max(all_x2) if all_x2 else 1000.0
    max_y  = max(all_y2) if all_y2 else 1000.0
    diagonal = math.hypot(max_x, max_y)

    # ── B. Duplicate grouping ──────────────────────────────────────────────
    if ablation == "no_duplicate":
        # Treat every detection as its own singleton group
        b_dup = {"groups": [{"group_id": f"g_{e['id']}", "members": [e["id"]],
                              "representative": e["id"], "reason": "singleton",
                              "is_singleton": True} for e in b_elems_raw],
                 "id_to_group": {e["id"]: f"g_{e['id']}" for e in b_elems_raw},
                 "representative": {f"g_{e['id']}": e["id"] for e in b_elems_raw},
                 "parent_child": []}
        a_dup = {"groups": [{"group_id": f"g_{e['id']}", "members": [e["id"]],
                              "representative": e["id"], "reason": "singleton",
                              "is_singleton": True} for e in a_elems_raw],
                 "id_to_group": {e["id"]: f"g_{e['id']}" for e in a_elems_raw},
                 "representative": {f"g_{e['id']}": e["id"] for e in a_elems_raw},
                 "parent_child": []}
        b_reps = b_elems_raw
        a_reps = a_elems_raw
    else:
        b_dup = build_duplicate_groups(b_elems_raw, cfg)
        a_dup = build_duplicate_groups(a_elems_raw, cfg)
        b_reps = group_representative_elems(b_elems_raw, b_dup)
        a_reps = group_representative_elems(a_elems_raw, a_dup)

    # ── C. OCR Anchors ────────────────────────────────────────────────────
    if ablation == "no_anchor":
        anchors = {"locked": [], "candidates": [], "skipped": [],
                   "anchor_shift": None}
    else:
        anchors = build_ocr_anchors(b_reps, a_reps, cfg)

    # ── D. Container matching ──────────────────────────────────────────────
    if ablation == "no_anchor":
        container_matches = []
    else:
        container_matches = match_containers(b_reps, a_reps, anchors, cfg)

    # Combine pre-matched IDs
    pre_b: set[str] = set()
    pre_a: set[str] = set()
    all_pre_records  = []

    for anc in anchors["locked"]:
        pre_b.add(anc["before_id"])
        pre_a.add(anc["after_id"])
        all_pre_records.append({
            "before_id":   anc["before_id"],
            "after_id":    anc["after_id"],
            "before_bbox": next((e["bbox"] for e in b_reps
                                  if e["id"]==anc["before_id"]), None),
            "after_bbox":  next((e["bbox"] for e in a_reps
                                  if e["id"]==anc["after_id"]), None),
            "match_type":  "ocr_anchor",
            "evidence":    anc,
        })

    anchors["container_candidates"] = container_matches

    # ── E/F. Remaining assignment ──────────────────────────────────────────
    cfg_assign = dict(cfg)
    if ablation == "no_crop":
        cfg_assign["crop_evidence_weight"] = 0.0

    # Use precomputed histograms; if no_crop set weight=0 so they are ignored
    assignment = solve_assignment(
        b_reps, a_reps, pre_b, pre_a, diagonal,
        anchors, cfg_assign,
        b_hists if ablation != "no_crop" else None,
        a_hists if ablation != "no_crop" else None,
    )

    # ── Merge all correspondence records ──────────────────────────────────
    id_to_b = {e["id"]: e for e in b_reps}
    id_to_a = {e["id"]: e for e in a_reps}

    def enrich(rec):
        """Add delta and classify flags to a correspondence record."""
        bid = rec.get("before_id")
        aid = rec.get("after_id")
        be = id_to_b.get(bid)
        ae = id_to_a.get(aid)
        if be and ae and rec.get("before_bbox") and rec.get("after_bbox"):
            delta = calc_delta(be, ae, diagonal)
            rec["delta"] = delta
            rec["before_source"] = be.get("source","")
            rec["after_source"]  = ae.get("source","")
            rec["before_class"] = be.get("ocr_text") or be.get("class_label", "")
            rec["after_class"] = ae.get("ocr_text") or ae.get("class_label", "")
            rec["change_reasons"] = [label for label, flag in [("center_shift",delta["center_shift"]>8),("edge_shift",delta["max_edge_shift"]>8),("relative_resize",max(abs(delta["relative_dw"]),abs(delta["relative_dh"]))>.15)] if flag]
            rec["quality_flags"] = []
            rec["before_group_members"] = be.get("group_members",[be["id"]])
            rec["after_group_members"] = ae.get("group_members",[ae["id"]])
            rec["before_original_bboxes"] = be.get("original_bboxes",{be["id"]:be["bbox"]})
            rec["after_original_bboxes"] = ae.get("original_bboxes",{ae["id"]:ae["bbox"]})
            rec["match_type"] = "container_child_anchors" if rec.get("evidence",{}).get("container_evidence") else rec["match_type"]
            if any(e["confidence"] is not None and e["confidence"] < cfg["anchor_min_confidence"] for e in (be,ae)): rec["quality_flags"].append("low_detection_confidence")
            if delta["center_shift"] > diagonal*.2 or max(abs(delta["relative_dw"]),abs(delta["relative_dh"])) > 1: rec["quality_flags"].append("large_motion_or_resize_review")
            if be["has_ocr"] and ae["has_ocr"] and be["ocr_text"]!=ae["ocr_text"]: rec["quality_flags"].append("possible_ocr_segmentation_or_content_change")
            for e, img in [(be,b_img),(ae,a_img)]:
                if img and (e["bbox"][0]<=0 or e["bbox"][1]<=0 or e["bbox"][2]>=img.width or e["bbox"][3]>=img.height): rec["quality_flags"].append("viewport_boundary_or_clipping")
            if be["has_ocr"] != ae["has_ocr"]: rec["quality_flags"].append("text_nontext_level_mismatch")
        return rec

    matched_pairs = [enrich(r) for r in
                     all_pre_records + assignment["matched_pairs"]]

    uncertain_pairs = [enrich(r) for r in assignment["uncertain_pairs"]]

    # ── Uniqueness check ───────────────────────────────────────────────────
    all_b_ids = [r["before_id"] for r in matched_pairs + uncertain_pairs]
    all_a_ids = [r["after_id"]  for r in matched_pairs + uncertain_pairs]
    assert len(all_b_ids) == len(set(all_b_ids)), \
        f"Duplicate before_id in matched: {pair_id}"
    assert len(all_a_ids) == len(set(all_a_ids)), \
        f"Duplicate after_id in matched: {pair_id}"

    for reps, used, unmatched in [(b_reps,all_b_ids,assignment["unmatched_before"]),(a_reps,all_a_ids,assignment["unmatched_after"])]:
        assert set(used).isdisjoint(e["id"] for e in unmatched)
        assert set(used) | {e["id"] for e in unmatched} == {e["id"] for e in reps}

    # ── Output ────────────────────────────────────────────────────────────
    result = {
        "pair_id":   pair_id,
        "ablation":  ablation,
        "config":    cfg,
        "summary": {
            "before_detections":    len(b_elems_raw),
            "after_detections":     len(a_elems_raw),
            "before_dup_groups":    len(b_dup["groups"]),
            "after_dup_groups":     len(a_dup["groups"]),
            "ocr_anchors_locked":   len(anchors["locked"]),
            "ocr_anchors_candidate":len(anchors["candidates"]),
            "container_candidates": len(container_matches),
            "container_matches": sum(bool(r.get("evidence",{}).get("container_evidence")) for r in matched_pairs),
            "matched_pairs":        len(matched_pairs),
            "uncertain_pairs":      len(uncertain_pairs),
            "unmatched_before":     len(assignment["unmatched_before"]),
            "unmatched_after":      len(assignment["unmatched_after"]),
            "anchor_shift":         anchors.get("anchor_shift"),
            "crop_available":       b_img is not None and a_img is not None,
        },
        "matched_pairs":    matched_pairs,
        "uncertain_pairs":  uncertain_pairs,
        "unmatched_before": [{"id": e["id"], "bbox": e["bbox"],
                               "source": e.get("source",""),
                               "ocr_text": e.get("ocr_text","")}
                              for e in assignment["unmatched_before"]],
        "unmatched_after":  [{"id": e["id"], "bbox": e["bbox"],
                               "source": e.get("source",""),
                               "ocr_text": e.get("ocr_text","")}
                              for e in assignment["unmatched_after"]],
        "evidence_detail": {
            "anchors":          anchors,
            "duplicate_before": b_dup,
            "duplicate_after":  a_dup,
            "container_matches": container_matches,
            "cost_matrix_info": assignment["cost_matrix_info"],
        },
        "image_paths": {k:str(v.resolve()) for k,v in image_paths.items()},
        "warnings": b_warn + a_warn + (["Crop images unavailable; appearance evidence missing"] if b_img is None or a_img is None else []),
    }

    (out_dir / f"matching_result_{ablation}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    , encoding='utf-8')

    return result


# ===========================================================================
# ABLATION RUNNER
# ===========================================================================

ABLATIONS = ["full", "no_duplicate", "no_anchor", "no_crop"]
CLASSIFIERS = ["relative15", "edges8"]


def run_all_ablations(pair_id: str, input_root: Path,
                      output_root: Path, cfg: dict) -> list[dict]:
    """
    Run all ablations for one pair, evaluate with all classifiers.
    Return rows for CSV comparison.
    """
    pair_dir = input_root / pair_id
    gt_path  = pair_dir / "groundtruth.json"
    rows = []

    for abl in cfg.get("run_ablations", ["full"]):
        try:
            result = process_pair(pair_dir, output_root, cfg, ablation=abl)
        except Exception as exc:
            print(f"[ERROR] {pair_id} ablation={abl}: {exc}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)
            raise

        all_matches = result["matched_pairs"]  # only locked/matched, not uncertain

        for clf in CLASSIFIERS:
            mode=cfg.get('coverage_mode','envelope')
            if mode in ('group_union','group_structural'):
                from geometry.prediction_groups import build_prediction_groups
                image_path=Path(result['image_paths']['before'])
                with Image.open(image_path) as image:image_size=image.size
                selected=[m for m in all_matches if classify_change(m['delta'],clf)]
                group_data=build_prediction_groups(selected,all_matches,image_size)
                (output_root/pair_id/f'groups_{abl}_{clf}.json').write_text(json.dumps(group_data,ensure_ascii=False,indent=2), encoding='utf-8')
                ev=evaluate_grouped(all_matches,gt_path,clf,group_data,'union_support' if mode=='group_union' else 'structural_proxy')
            else:
                ev = evaluate(all_matches, gt_path, clf)
            # Save per-ablation eval
            eval_path = (output_root / pair_id
                         / f"eval_{abl}_{clf}.json")
            eval_path.write_text(json.dumps(ev, indent=2), encoding='utf-8')

            row = {
                "pair_id":        pair_id,
                "ablation":       abl,
                "classifier":     clf,
                "matched_pairs":  len(all_matches),
                "uncertain_pairs":len(result["uncertain_pairs"]),
                "unmatched_before":len(result["unmatched_before"]),
                "unmatched_after": len(result["unmatched_after"]),
                "ocr_anchors":    result["summary"]["ocr_anchors_locked"],
                "container_matches": result["summary"]["container_matches"],
                "group_count": ev.get('group_count',0),
                "group_rescued": len(ev.get('group_rescued_gt',[])),
                "gt_count":       ev["gt_count"],
                "covered":        ev["covered"],
                "predictions":    ev["predictions"],
                "unverified":     ev["unverified"],
                "pass":           ev["pass"],
                "missed_gt":      ",".join(str(i) for i in ev["missed_gt"]),
            }
            rows.append(row)

    return rows


# ===========================================================================
# VISUALIZATION (bbox overlay)
# ===========================================================================

def visualize_pair(pair_id: str, pair_dir: Path, output_dir: Path,
                   new_result: dict, gt_path: Path,
                   classifier: str = "edges8") -> Optional[str]:
    """
    Draw GT (red), new matched predictions (green), uncertain (yellow)
    on before and after images. Returns output path string or None.
    """
    if not _PIL_OK:
        return None

    gt_data = json.loads(gt_path.read_text(encoding='utf-8'))
    gt_bboxes = [g for g in gt_data.get("bboxes", [])
                 if "geometry" in g.get("type", [])]

    # Filter predictions by classifier
    all_matches = new_result.get("matched_pairs", [])
    selected = [m for m in all_matches
                if "delta" in m and classify_change(m["delta"], classifier)]

    try:
        from PIL import ImageDraw, ImageFont
        font = None
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except Exception:
            pass

        def draw_overlay(img_path, side, out_path):
            if not img_path.exists():
                return
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            w, h = img.size

            # GT boxes — red solid
            for gi, g in enumerate(gt_bboxes):
                x1,y1 = g["x"],g["y"]
                x2,y2 = x1+g["w"], y1+g["h"]
                draw.rectangle([x1,y1,x2,y2], outline="#ff3333", width=3)
                lbl = f"GT{gi}"
                if font:
                    draw.text((max(0,x1), max(0,y1-16)), lbl, fill="#ff3333", font=font)

            # Predictions — green dashed
            for pi, m in enumerate(selected):
                bbox_key = f"{side}_bbox"
                bb = m.get(bbox_key)
                if not bb:
                    continue
                x1,y1,x2,y2 = [int(v) for v in bb]
                draw.rectangle([x1,y1,x2,y2], outline="#22ee55", width=2)
                lbl = f"P{pi} {m['before_id']}->{m['after_id']}"
                if font:
                    draw.text((max(0,x1), max(0,y1-16)), lbl, fill="#22ee55", font=font)

            # Uncertain — yellow
            for m in new_result.get("uncertain_pairs", []):
                bbox_key = f"{side}_bbox"
                bb = m.get(bbox_key)
                if not bb:
                    continue
                x1,y1,x2,y2 = [int(v) for v in bb]
                draw.rectangle([x1,y1,x2,y2], outline="#ffdd00", width=2)
                if font: draw.text((max(0,x1),max(0,y1-16)),f"UNCERTAIN {m['before_id']}->{m['after_id']}",fill="#ffdd00",font=font)

            img.save(str(out_path))

        out_before = output_dir / f"{pair_id}_before_overlay.png"
        out_after  = output_dir / f"{pair_id}_after_overlay.png"
        draw_overlay(Path(new_result["image_paths"]["before"]), "before", out_before)
        draw_overlay(pair_dir / "after.png",  "after",  out_after)
        return str(output_dir)

    except Exception as exc:
        print(f"[WARN] Visualization failed for {pair_id}: {exc}", file=sys.stderr)
        return None


# ===========================================================================
# CLI MAIN
# ===========================================================================

def main():
    import csv
    parser = argparse.ArgumentParser(description="Evidence-based geometry matching")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--pair-ids", nargs="+")
    selection.add_argument("--all-geometry", action="store_true", help="Explicitly run all numeric pairs containing geometry GT")
    parser.add_argument("--image-root", help="Directory containing <pair_id>_* folders with baseline.png/inject.png")
    parser.add_argument("--ablations", action="store_true", help="Run component ablations as well as full matching")
    parser.add_argument("--coverage-mode",choices=["envelope","group_union","group_structural"],default="group_union",help="group_union uses exact member-union support; structural proxy is experimental")
    args = parser.parse_args()
    start = time.perf_counter()
    input_root, output_root = Path(args.input_root).resolve(), Path(args.output_root).resolve()
    if not input_root.is_dir(): parser.error("Input directory does not exist")
    if output_root == input_root or input_root.is_relative_to(output_root):
        parser.error("Output must be separate from input and cannot be its ancestor")
    if args.all_geometry:
        # Dataset selection only. These labels are never passed to the matcher.
        pair_ids = []
        for folder in sorted(input_root.iterdir()):
            if folder.is_dir() and folder.name.isdigit() and (folder/"groundtruth.json").is_file():
                data = json.loads((folder/"groundtruth.json").read_text(encoding='utf-8'))
                if any("geometry" in g.get("type",[]) for g in data.get("bboxes",[])): pair_ids.append(folder.name)
    else:
        pair_ids = args.pair_ids
    if not pair_ids or len(set(pair_ids)) != len(pair_ids) or any(not pid.isdigit() for pid in pair_ids):
        parser.error("Pair IDs must be nonempty, unique numeric directory names")
    image_root = str(Path(args.image_root).resolve()) if args.image_root else None
    provenance = []
    for pid in pair_ids:
        folder = input_root/pid
        if (output_root/pid).resolve() == folder.resolve(): parser.error("Output aliases input")
        for name in ("before.json","after.json","groundtruth.json"):
            path = folder/name
            if not path.is_file(): parser.error(f"Missing {path}")
            provenance.append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        for path in resolve_images(folder,image_root).values():
            if not path.is_file(): parser.error(f"Missing image: {path}; supply --image-root")
            provenance.append({"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CONFIG, pipeline_version="evidence-v3-20261016", evaluation_protocol=(EVALUATION_PROTOCOL if args.coverage_mode=="envelope" else "group_coverage_v1_"+("union_support" if args.coverage_mode=="group_union" else "structural_proxy")), coverage_mode=args.coverage_mode, image_root=image_root,
               run_ablations=ABLATIONS if args.ablations else ["full"])
    (output_root/"config.json").write_text(json.dumps(cfg,indent=2), encoding='utf-8')
    (output_root/"run_status.json").write_text(json.dumps({"status":"running","requested_pairs":pair_ids}), encoding='utf-8')
    print(f"Selected {len(pair_ids)} pairs; output={output_root}",flush=True)
    all_rows = []
    try:
        for idx,pid in enumerate(pair_ids,1):
            rows = run_all_ablations(pid,input_root,output_root,cfg)
            all_rows.extend(rows)
            result = json.loads((output_root/pid/"matching_result_full.json").read_text(encoding='utf-8'))
            visualize_pair(pid,input_root/pid,output_root/pid,result,input_root/pid/"groundtruth.json","edges8")
            row = next(r for r in rows if r["ablation"]=="full" and r["classifier"]=="edges8")
            print(f"[{idx}/{len(pair_ids)}] {pid} GT={row['covered']}/{row['gt_count']} PASS={row['pass']} predictions={row['predictions']} uncertain={row['uncertain_pairs']}",flush=True)
            with (output_root/"evaluation.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    except Exception as exc:
        (output_root/"run_status.json").write_text(json.dumps({"status":"failed","error":str(exc),"completed_pairs":sorted({r["pair_id"] for r in all_rows}),"requested_pairs":pair_ids},indent=2), encoding='utf-8')
        raise
    for entry in provenance:
        if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()!=entry["sha256"]:
            raise RuntimeError(f"Input changed during run: {entry['path']}")
    summary={}
    for clf in CLASSIFIERS:
        rs=[r for r in all_rows if r["ablation"]=="full" and r["classifier"]==clf]
        summary[clf]={"samples":len(rs),"pass_samples":sum(r["pass"] for r in rs),"gt_count":sum(r["gt_count"] for r in rs),"covered":sum(r["covered"] for r in rs),"predictions":sum(r["predictions"] for r in rs),"unverified":sum(r["unverified"] for r in rs),"matched_pairs":sum(r["matched_pairs"] for r in rs),"uncertain_pairs":sum(r["uncertain_pairs"] for r in rs)}
    manifest={"pipeline_version":cfg["pipeline_version"],"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"pair_ids":pair_ids,"config":cfg,"elapsed_seconds":time.perf_counter()-start,"inputs":provenance,"evaluation_policy":"matched_only","evaluation_protocol":cfg["evaluation_protocol"],"input_hashes_verified":True,"summary":summary}
    (output_root/"run_manifest.json").write_text(json.dumps(manifest,indent=2), encoding='utf-8')
    (output_root/"matching_manifest.json").write_text(json.dumps({k:manifest[k] for k in ("pipeline_version","pair_ids","config","inputs","elapsed_seconds")},indent=2), encoding='utf-8')
    (output_root/"summary.json").write_text(json.dumps(summary,indent=2), encoding='utf-8')
    (output_root/"run_status.json").write_text(json.dumps({"status":"complete","completed_pairs":pair_ids},indent=2), encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
