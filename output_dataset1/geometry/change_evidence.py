"""GT-independent visual motion checks; containment event coverage is a separate metric."""
import math
import numpy as np
from PIL import Image
from shared.regions import area,intersect,union_area
CONFIG={'crop_size':32,'min_visible_fraction':.9,'min_crop_area':64,'min_texture_std':8.,'min_aligned_similarity':.80,'min_alignment_improvement':.02,'min_changed_fraction':.01,'pixel_channel_difference':24,'min_group_members':2,'min_group_containment':.90}
def crop_array(image,bbox):
    bb=(math.floor(bbox[0]),math.floor(bbox[1]),math.ceil(bbox[2]),math.ceil(bbox[3]))
    return np.asarray(image.crop(bb).resize((CONFIG['crop_size'],CONFIG['crop_size']),Image.Resampling.BILINEAR).convert('L'),dtype=np.float32)
def motion_features(before,after,record):
    b=record['before_bbox'];a=record['after_bbox'];fraction=min(area(intersect(b,[0,0,before.width,before.height]))/max(area(b),1e-9),area(intersect(a,[0,0,after.width,after.height]))/max(area(a),1e-9))
    if fraction<CONFIG['min_visible_fraction'] or min(area(b),area(a))<CONFIG['min_crop_area']:return {'strong':False,'reason':'crop_clipped_or_too_small','visible_fraction':fraction}
    bc=crop_array(before,b);ac=crop_array(after,a);sa=crop_array(after,b);sb=crop_array(before,a)
    aligned=float(np.abs(bc-ac).mean()/255);stationary=float((np.abs(bc-sa).mean()+np.abs(ac-sb).mean())/(2*255));difference=max(float((np.abs(bc-sa)>CONFIG['pixel_channel_difference']).mean()),float((np.abs(ac-sb)>CONFIG['pixel_channel_difference']).mean()));texture=float(min(bc.std(),ac.std()))
    anchor=record.get('match_type')=='ocr_anchor' and bool(record.get('before_class')) and record.get('before_class')==record.get('after_class');stable=record.get('evidence',{}).get('cost_gap_ban',0)>=.05
    strong=bool((anchor or stable) and 'low_detection_confidence' not in record.get('quality_flags',[]) and texture>=CONFIG['min_texture_std'] and 1-aligned>=CONFIG['min_aligned_similarity'] and stationary-aligned>=CONFIG['min_alignment_improvement'] and difference>=CONFIG['min_changed_fraction'])
    return {'strong':strong,'aligned_similarity':1-aligned,'stationary_similarity':1-stationary,'alignment_improvement':stationary-aligned,'changed_fraction':difference,'texture_std':texture,'visible_fraction':fraction,'exact_ocr_identity':bool(anchor),'stable_assignment':bool(stable)}
def contained_group_metrics(regions,gt_box):
    support=union_area(regions);inter=union_area([intersect(b,gt_box) for b in regions])
    return {'prediction_containment':inter/max(support,1e-9),'gt_fraction':inter/max(area(gt_box),1e-9),'iou':inter/max(support+area(gt_box)-inter,1e-9)}
def event_links(groups,gt_box):
    links=[]
    for g in groups:
        if len(g['members'])<CONFIG['min_group_members'] or not all(m.get('visual_evidence',{}).get('strong') for m in g['members']):continue
        value=contained_group_metrics(g['member_regions'],gt_box)
        if value['prediction_containment']+1e-12>=CONFIG['min_group_containment']:links.append({'group_id':g['group_id'],'member_prediction_indices':g['original_prediction_indices'],'member_count':len(g['members']),**value,'human_semantic_correct':None})
    return links
