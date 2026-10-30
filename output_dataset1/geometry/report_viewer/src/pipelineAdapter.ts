import type { MatchingResult, MatchItem, Geometry } from './types';

type RawPair = {
  before_id: string; after_id: string; before_bbox: [number,number,number,number]; after_bbox: [number,number,number,number];
  delta: MatchItem['delta']; before_source?: string; after_source?: string; before_class?: string; after_class?: string;
  match_type: string; evidence: Record<string, unknown>; change_reasons?: string[]; quality_flags?: string[];
};
type RawResult = {
  pair_id: string; matched_pairs: RawPair[]; uncertain_pairs: RawPair[];
  summary: { before_detections: number; after_detections: number; unmatched_before: number; unmatched_after: number };
  unmatched_before: { id: string; bbox: [number,number,number,number]; source: string; ocr_text?: string }[];
  unmatched_after: RawResult['unmatched_before'];
};
function geometry(b: RawPair['before_bbox']): Geometry {
  return { cx:(b[0]+b[2])/2,cy:(b[1]+b[3])/2,width:b[2]-b[0],height:b[3]-b[1] };
}
function iou(a: RawPair['before_bbox'], b: RawPair['before_bbox']): number {
  const intersection = Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0])) * Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
  return intersection / ((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection);
}
export function adaptResult(raw: RawResult, includeUncertain = false): MatchingResult {
  const convert = (m: RawPair, uncertain: boolean): MatchItem => ({
    match_type: uncertain ? 'uncertain' : m.match_type,
    before_id:m.before_id,after_id:m.after_id,before_source:m.before_source ?? '',after_source:m.after_source ?? '',
    before_class:m.before_class ?? '',after_class:m.after_class ?? '',before_bbox:m.before_bbox,after_bbox:m.after_bbox,
    before_geometry:geometry(m.before_bbox),after_geometry:geometry(m.after_bbox),delta:m.delta,
    matching_features: { position_distance:Math.hypot(m.delta.dx,m.delta.dy),size_difference:Math.max(Math.abs(m.delta.relative_dw),Math.abs(m.delta.relative_dh)),aspect_difference:Math.abs(Math.log((geometry(m.after_bbox).width/geometry(m.after_bbox).height)/(geometry(m.before_bbox).width/geometry(m.before_bbox).height))),iou:iou(m.before_bbox,m.after_bbox),text_similarity:null,same_source:m.before_source===m.after_source },
    match_cost:Number(m.evidence.cost ?? 0),ambiguity_margin:typeof m.evidence.margin==='number' ? m.evidence.margin : null,
    description:[...(uncertain ? ['UNCERTAIN — chưa xác nhận'] : []),...(m.change_reasons ?? []),...(m.quality_flags ?? [])],
  });
  const matches = raw.matched_pairs.map(m => convert(m,false));
  if (includeUncertain) matches.push(...raw.uncertain_pairs.map(m => convert(m,true)));
  return {
    pair_id:raw.pair_id, matches,
    summary:{ before_elements:raw.summary.before_detections,after_elements:raw.summary.after_detections,matched_pairs:raw.matched_pairs.length,uncertain_pairs:raw.uncertain_pairs.length,unmatched_before:raw.summary.unmatched_before,unmatched_after:raw.summary.unmatched_after,evidence_breakdown:{ocr_anchors:raw.matched_pairs.filter(m=>m.match_type==='ocr_anchor').length,container_matches:raw.matched_pairs.filter(m=>m.evidence.container_evidence).length,evidence_assignments:raw.matched_pairs.filter(m=>m.match_type==='evidence_assignment').length} },
    unmatched_before:raw.unmatched_before.map(m=>({...m,class_label:m.ocr_text ?? '',side:'before'})),
    unmatched_after:raw.unmatched_after.map(m=>({...m,class_label:m.ocr_text ?? '',side:'after'})),
  };
}
