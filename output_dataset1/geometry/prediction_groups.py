"""GT-independent grouping of changed correspondences. No GT/file access."""
import math
from statistics import median

GROUP_CONFIG = {
    'min_members': 2,
    'min_structural_members': 3,
    'max_image_area_ratio': .35,
    'parent_containment': .85,
    'max_neighbour_gap_px': 64.,
    'motion_tolerance_px': 12.,
    'motion_relative_tolerance': .25,
    'structural_min_support_ccr': .10,
    'structural_min_spread': .50,
}

from shared.regions import area, intersect, enclosure, inside, iou, union_area

def motion_compatible(a,b,cfg):
    da,db=a['delta'],b['delta']
    tolerance=max(cfg['motion_tolerance_px'],cfg['motion_relative_tolerance']*max(math.hypot(da['dx'],da['dy']),math.hypot(db['dx'],db['dy'])))
    return math.hypot(da['dx']-db['dx'],da['dy']-db['dy'])<=tolerance

def neighbours(a,b,cfg):
    for side in ('before_bbox','after_bbox'):
        x,y=a[side],b[side]
        gap=math.hypot(max(0,x[0]-y[2],y[0]-x[2]),max(0,x[1]-y[3],y[1]-x[3]))
        if gap>cfg['max_neighbour_gap_px']:return False
    return True

def build_prediction_groups(selected,all_records,image_size,config=None):
    """Use matched containers and coherent local motion, never GT geometry.

    selected order is the prediction_index order used by envelope evaluation.
    Parent groups use smallest containing matched container at BOTH endpoints.
    Motion groups require complete-link consistency, avoiding chain drift.
    """
    cfg=dict(GROUP_CONFIG,**(config or {}));w,h=image_size
    if w<=0 or h<=0:raise ValueError('Invalid image size')
    parents=[m for m in all_records if m.get('before_source')!='ocr' and m.get('after_source')!='ocr' and not ('text_nontext_level_mismatch' in m.get('quality_flags',[]))]
    parent_members={};parent_lookup={}
    for i,m in enumerate(selected):
        eligible=[]
        for parent in parents:
            if parent['before_id']==m['before_id'] or parent['after_id']==m['after_id']:continue
            if all(inside(m[s+'_bbox'],parent[s+'_bbox'])>=cfg['parent_containment'] and area(parent[s+'_bbox'])>1.25*area(m[s+'_bbox']) for s in ('before','after')):
                eligible.append(parent)
        if eligible:
            parent=min(eligible,key=lambda p:area(p['before_bbox'])+area(p['after_bbox']))
            key=(parent['before_id'],parent['after_id']);parent_members.setdefault(key,[]).append(i);parent_lookup[key]=parent
    candidates=[]
    for key,ids in sorted(parent_members.items()):candidates.append(('shared_parent',ids,parent_lookup[key]))
    # Greedy deterministic clusters; each newly added pair must agree with all
    # existing members in motion and touch the local cluster in BOTH images.
    for seed in range(len(selected)):
        ids=[seed]
        for i in range(len(selected)):
            if i in ids:continue
            if all(motion_compatible(selected[i],selected[j],cfg) for j in ids) and any(neighbours(selected[i],selected[j],cfg) for j in ids):ids.append(i)
        candidates.append(('local_motion',sorted(ids),None))
    groups=[];seen=set()
    for kind,ids,parent in candidates:
        # Keep one representative for duplicate text/visual evidence at both ends.
        unique=[]
        for i in sorted(ids,key=lambda i:(-area(selected[i]['before_bbox']),i)):
            if not any(all(iou(selected[i][s+'_bbox'],selected[j][s+'_bbox'])>=.7 for s in ('before','after')) for j in unique):unique.append(i)
        ids=sorted(unique)
        if len(ids)<cfg['min_members']:continue
        before=enclosure([selected[i]['before_bbox'] for i in ids]);after=enclosure([selected[i]['after_bbox'] for i in ids]);region=enclosure([before,after])
        if area(region)/(w*h)>cfg['max_image_area_ratio']:continue
        member_regions=[enclosure([selected[i]['before_bbox'],selected[i]['after_bbox']]) for i in ids]
        signature=(tuple(ids),tuple(region))
        if signature in seen:continue
        seen.add(signature)
        cx=[(b[0]+b[2])/2 for b in member_regions];cy=[(b[1]+b[3])/2 for b in member_regions]
        spread=max((max(cx)-min(cx))/max(region[2]-region[0],1),(max(cy)-min(cy))/max(region[3]-region[1],1))
        support_area=union_area(member_regions)
        groups.append({'group_id':f'group_{len(groups):03d}','kind':kind,'member_prediction_indices':ids,
                       'members':[{'prediction_index':i,'before_id':selected[i]['before_id'],'after_id':selected[i]['after_id'],'before_bbox':selected[i]['before_bbox'],'after_bbox':selected[i]['after_bbox'],'delta':selected[i]['delta']} for i in ids],
                       'before_bbox':before,'after_bbox':after,'prediction_region':region,'member_regions':member_regions,
                       'support_area':support_area,'bbox_area':area(region),'support_density':support_area/max(area(region),1),
                       'image_area_ratio':area(region)/(w*h),'spatial_spread':spread,
                       'parent':{'before_id':parent['before_id'],'after_id':parent['after_id'],'before_bbox':parent['before_bbox'],'after_bbox':parent['after_bbox']} if parent else None,
                       'all_motion_compatible':all(motion_compatible(selected[i],selected[j],cfg) for i in ids for j in ids),
                       'needs_human_review':True})
    return {'config':cfg,'image_size':list(image_size),'selected_predictions':len(selected),'groups':groups,'uses_groundtruth':False}
