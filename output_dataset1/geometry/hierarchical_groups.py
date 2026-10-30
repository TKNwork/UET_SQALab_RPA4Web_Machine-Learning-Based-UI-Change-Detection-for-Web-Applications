"""GT-independent multi-level groups and explicitly inferred layout regions.

Neither grouping nor proposal generation receives GT. Inferred extents are NOT
member union support and must be reported/evaluated as a separate evidence type.
"""
import math
from statistics import median
from geometry.prediction_groups import GROUP_CONFIG,motion_compatible,neighbours
from shared.regions import area,inside,iou,enclosure,union_area

CONFIG={
 'parent_containment':.85,'max_image_area_ratio':.35,'min_union_members':2,
 'min_layout_members':3,'min_exact_ocr_anchors':2,'min_spatial_spread':.5,
 'max_adaptive_gap_px':160.,'adaptive_gap_height_factor':3.,
 'min_pixel_change_fraction':.01,'pixel_channel_difference':24,
 'min_parent_extent_area_fraction':.5,
}

def unique_members(ids,selected):
    result=[]
    for i in sorted(ids,key=lambda i:(-area(selected[i]['before_bbox']),i)):
        if not any(all(iou(selected[i][s+'_bbox'],selected[j][s+'_bbox'])>=.7 for s in ('before','after')) for j in result):result.append(i)
    return sorted(result)

def coherent_clusters(ids,selected):
    result=[]
    for seed in ids:
        group=[seed]
        for i in ids:
            if i not in group and all(motion_compatible(selected[i],selected[j],GROUP_CONFIG) for j in group):group.append(i)
        result.append(sorted(group))
    return result

def build_hierarchical(selected,all_records,image_size,baseline_groups,pixel_scorer=None,config=None):
    cfg=dict(CONFIG,**(config or {}));w,h=image_size
    if w<=0 or h<=0:raise ValueError('Invalid image dimensions')
    groups=list(baseline_groups['groups']);proposals=[];seen={tuple(g['member_prediction_indices']) for g in groups};proposal_seen=set();candidates=[]
    parents=[m for m in all_records if m.get('before_source')!='ocr' and m.get('after_source')!='ocr' and 'text_nontext_level_mismatch' not in m.get('quality_flags',[])]
    # Retain children at EVERY plausible ancestor, instead of the smallest only.
    for parent in parents:
        ids=[i for i,m in enumerate(selected) if m['before_id']!=parent['before_id'] and m['after_id']!=parent['after_id'] and all(inside(m[s+'_bbox'],parent[s+'_bbox'])>=cfg['parent_containment'] and area(parent[s+'_bbox'])>1.25*area(m[s+'_bbox']) for s in ('before','after'))]
        for cluster in coherent_clusters(ids,selected):candidates.append(('ancestor_motion',cluster,parent))
    # Sparse UI layouts may have gaps larger than 64 px; bound adaptive gaps.
    gap=max(GROUP_CONFIG['max_neighbour_gap_px'],min(cfg['max_adaptive_gap_px'],cfg['adaptive_gap_height_factor']*median([m['before_bbox'][3]-m['before_bbox'][1] for m in selected]))) if selected else 64
    localcfg=dict(GROUP_CONFIG,max_neighbour_gap_px=gap)
    for seed in range(len(selected)):
        ids=[seed]
        for i in range(len(selected)):
            if i not in ids and all(motion_compatible(selected[i],selected[j],GROUP_CONFIG) for j in ids) and any(neighbours(selected[i],selected[j],localcfg) for j in ids):ids.append(i)
        candidates.append(('adaptive_local_motion',sorted(ids),None))
    # Existing groups also receive the same guards for the inferred extent branch.
    candidates += [(g['kind'],g['member_prediction_indices'],g.get('parent')) for g in baseline_groups['groups']]
    for kind,raw_ids,parent in candidates:
        ids=unique_members(raw_ids,selected)
        if len(ids)<cfg['min_union_members']:continue
        if not all(motion_compatible(selected[i],selected[j],GROUP_CONFIG) for i in ids for j in ids):continue
        members=[{'prediction_index':i,**{k:selected[i][k] for k in ('before_id','after_id','before_bbox','after_bbox','delta')}} for i in ids]
        before=enclosure([m['before_bbox'] for m in members]);after=enclosure([m['after_bbox'] for m in members]);extent=enclosure([before,after])
        if area(extent)/(w*h)>cfg['max_image_area_ratio']:continue
        regions=[enclosure([m['before_bbox'],m['after_bbox']]) for m in members]
        cx=[(b[0]+b[2])/2 for b in regions];cy=[(b[1]+b[3])/2 for b in regions]
        spread=max((max(cx)-min(cx))/max(extent[2]-extent[0],1),(max(cy)-min(cy))/max(extent[3]-extent[1],1))
        group={'group_id':f'hierarchy_{len(groups):04d}','kind':kind,'member_prediction_indices':ids,'members':members,'before_bbox':before,'after_bbox':after,'prediction_region':extent,'member_regions':regions,'support_area':union_area(regions),'bbox_area':area(extent),'support_density':union_area(regions)/max(area(extent),1),'image_area_ratio':area(extent)/(w*h),'spatial_spread':spread,'parent':parent,'all_motion_compatible':True,'needs_human_review':True}
        if tuple(ids) not in seen:groups.append(group);seen.add(tuple(ids))
        anchors=sum(bool(selected[i].get('match_type')=='ocr_anchor' and selected[i].get('before_class') and selected[i].get('before_class')==selected[i].get('after_class')) for i in ids)
        if len(ids)<cfg['min_layout_members'] or anchors<cfg['min_exact_ocr_anchors'] or spread<cfg['min_spatial_spread'] or pixel_scorer is None:continue
        region_candidates=[('coherent_layout_extent',extent)]
        if parent and all(k in parent for k in ('before_bbox','after_bbox')):
            pregion=enclosure([parent['before_bbox'],parent['after_bbox']])
            if all(area(b)/max(area(parent[s+'_bbox']),1)>=cfg['min_parent_extent_area_fraction'] for s,b in [('before',before),('after',after)]):region_candidates.append(('supported_container_extent',pregion))
        for evidence_type,region in region_candidates:
            signature=(tuple(ids),tuple(region))
            if signature in proposal_seen or area(region)/(w*h)>cfg['max_image_area_ratio']:continue
            pixel=pixel_scorer(region)
            if pixel.get('visible_fraction',0)<.9 or pixel.get('change_fraction',0)<cfg['min_pixel_change_fraction']:continue
            proposal_seen.add(signature)
            proposals.append({'proposal_id':f'layout_{len(proposals):04d}','evidence_type':evidence_type,'region':region,'member_prediction_indices':ids,'members':members,'member_regions':regions,'parent':parent,'exact_ocr_anchor_count':anchors,'spatial_spread':spread,'pixel_evidence':pixel,'inferred_region_not_union_support':True,'needs_human_review':True})
    return {'config':dict(baseline_groups['config']),'hierarchy_config':cfg,'image_size':list(image_size),'selected_predictions':len(selected),'groups':groups,'layout_proposals':proposals,'uses_groundtruth':False}
