"""Geometry coverage against native side-specific GT, without GT endpoint pairing.

Units are annotated endpoints, not inferred before/after objects or envelopes.
Only already-selected changed matched correspondences and frozen groups are used.
"""
import math
from shared.regions import area,intersect,union_area
from geometry.pipeline import classify_change,prediction_envelope

PROTOCOL='geometry_native_gt_envelope_coverage_v1'
THRESHOLDS={'iou':0.30,'ccr':0.50}

def support_metrics(boxes,gt):
    intersection=union_area([intersect(b,gt) for b in boxes])
    return {'iou':intersection/max(area(gt)+union_area(boxes)-intersection,1e-9),
            'ccr':intersection/max(area(gt),1e-9)}

def evaluate_native(matched,gt,groups,classifier='edges8',mode='union_support'):
    if mode not in ('single_envelope','union_support'):raise ValueError(mode)
    if not all(isinstance(gt.get(s+'_bboxes'),list) for s in ('before','after')):
        raise ValueError('Native GT must contain before_bboxes and after_bboxes lists')
    selected=[m for m in matched if 'delta' in m and classify_change(m['delta'],classifier)]
    if groups.get('uses_groundtruth') is not False or groups['selected_predictions']!=len(selected):
        raise ValueError('Groups do not correspond to frozen predictions')
    for group in groups['groups']:
        if not group['members']:raise ValueError('Empty frozen group')
        indices=[]
        for member in group['members']:
            i=member['prediction_index']
            if not isinstance(i,int) or not 0<=i<len(selected):raise ValueError('Invalid member index')
            if any(member[k]!=selected[i][k] for k in ('before_id','after_id','before_bbox','after_bbox')):
                raise ValueError('Frozen group member differs from correspondence')
            indices.append(i)
        if indices!=group['member_prediction_indices']:raise ValueError('Member index mismatch')
    coverage=[];used=set();per_side={}
    for side in ('before','after'):
        supports=[{'kind':'single','prediction_indices':[i],'boxes':[prediction_envelope(m['before_bbox'],m['after_bbox'])]} for i,m in enumerate(selected)]
        if mode=='union_support':
            supports += [{'kind':'group','group_id':g['group_id'],'prediction_indices':g['member_prediction_indices'],
                          'boxes':[prediction_envelope(m['before_bbox'],m['after_bbox']) for m in g['members']]} for g in groups['groups']]
        start=len(coverage)
        for index,g in enumerate(gt[side+'_bboxes']):
            if 'geometry' not in g.get('type',[]):continue
            values=[g.get(k) for k in ('x','y','w','h')]
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values) or g['w']<=0 or g['h']<=0:
                raise ValueError(f'Invalid {side} geometry bbox {index}')
            box=[g['x'],g['y'],g['x']+g['w'],g['y']+g['h']];links=[]
            for support in supports:
                metrics=support_metrics(support['boxes'],box)
                if metrics['iou']+1e-12<THRESHOLDS['iou'] and metrics['ccr']+1e-12<THRESHOLDS['ccr']:continue
                actual=[i for i,b in zip(support['prediction_indices'],support['boxes']) if area(intersect(b,box))>0]
                endpoint_intersection=sum(area(intersect(selected[i][side+'_bbox'],box)) for i in actual)
                links.append({k:v for k,v in support.items() if k!='boxes'}|metrics|{'overlapping_prediction_indices':actual,'same_side_endpoint_intersection_area_sum':endpoint_intersection,'envelope_gap_only_on_gt_side':endpoint_intersection==0})
                used.update(actual)
            coverage.append({'side':side,'original_gt_index':index,'selector':g.get('selector'),'gt':g,'covered':bool(links),'links':links})
        side_rows=coverage[start:]
        per_side[side]={'gt_count':len(side_rows),'covered':sum(r['covered'] for r in side_rows)}
    if not coverage:raise ValueError('No geometry GT; cannot classify as PASS')
    return {'protocol':PROTOCOL,'classifier':classifier,'mode':mode,'thresholds':THRESHOLDS,
            'unit':'side-specific original geometry GT annotation; no inferred GT identity pairing',
            'coverage_rule':'singleton before-after envelope OR exact union of member envelopes of one frozen group; IoU>=0.30 OR CCR>=0.50',
            'gt_count':len(coverage),'covered':sum(r['covered'] for r in coverage),'per_side':per_side,
            'pass':all(r['covered'] for r in coverage),'predictions':len(selected),'groups':len(groups['groups']) if mode=='union_support' else 0,
            'supporting_predictions':len(used),'unverified':len(selected)-len(used),'gt_coverage':coverage,
            'missed_gt':[{'side':r['side'],'original_gt_index':r['original_gt_index'],'selector':r['selector']} for r in coverage if not r['covered']]}
