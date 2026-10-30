export interface BBox {
  0: number; // x1
  1: number; // y1
  2: number; // x2
  3: number; // y2
}

export interface Geometry {
  cx: number;
  cy: number;
  width: number;
  height: number;
}

export interface Delta {
  max_edge_shift?: number;
  dx: number;
  dy: number;
  dw: number;
  dh: number;
  relative_dx: number;
  relative_dy: number;
  relative_dw: number;
  relative_dh: number;
}

export interface MatchingFeatures {
  position_distance: number;
  size_difference: number;
  aspect_difference: number;
  iou: number;
  text_similarity: number | null;
  same_source: boolean;
}

export interface MatchItem {
  match_type: "ocr_anchor" | "container_child_anchors" | "evidence_assignment" | string;
  before_id: string;
  after_id: string;
  before_source: string;
  after_source: string;
  before_class: string;
  after_class: string;
  before_bbox: [number, number, number, number];
  after_bbox: [number, number, number, number];
  before_geometry: Geometry;
  after_geometry: Geometry;
  delta: Delta;
  matching_features: MatchingFeatures;
  match_cost: number;
  ambiguity_margin: number | null;
  description: string[];
}

export interface UnmatchedItem {
  id: string;
  source: string;
  class_label: string;
  bbox: [number, number, number, number];
  side: "before" | "after";
}

export interface EvidenceBreakdown {
  ocr_anchors: number;
  container_matches: number;
  evidence_assignments: number;
  no_description_diagnostics?: number;
}

export interface Summary {
  uncertain_pairs?: number;
  before_elements: number;
  after_elements: number;
  matched_pairs: number;
  unmatched_before: number;
  unmatched_after: number;
  evidence_breakdown: EvidenceBreakdown;
  invalid_before_detections?: number;
  invalid_after_detections?: number;
}

export interface MatchingResult {
  pair_id: string;
  summary: Summary;
  matches: MatchItem[];
  unmatched_before: UnmatchedItem[];
  unmatched_after: UnmatchedItem[];
}

export type ViewMode = "side-by-side" | "overlay";

export type FilterPreset = "edges8" | "all" | "shift_gt_8" | "size_gt_15" | "unmatched" | "custom";

export type SortField = "d_position" | "d_size" | "iou" | "match_cost" | "id";
export type SortOrder = "asc" | "desc";

export interface FilterState {
  preset: FilterPreset;
  minShiftPx: number;
  minSizePct: number;
  tier: "all" | "ocr_anchor" | "container_child_anchors" | "evidence_assignment";
  searchQuery: string;
  hideMinorChanges: boolean; // hide if d_pos < 0.5px and d_size < 1%
  showUnmatchedBefore: boolean;
  showUnmatchedAfter: boolean;
}
