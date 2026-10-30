import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from geometry import pipeline as pipeline
from geometry.prediction_groups import build_prediction_groups,union_area


def record(i,box,dx=15,dy=0):
    after=[box[0]+dx,box[1]+dy,box[2]+dx,box[3]+dy]
    def e(b):return {'bbox':b,'cx':(b[0]+b[2])/2,'cy':(b[1]+b[3])/2,'width':b[2]-b[0],'height':b[3]-b[1]}
    return {'before_id':f'b{i}','after_id':f'a{i}','before_bbox':box,'after_bbox':after,'before_source':'ocr','after_source':'ocr','delta':pipeline.calc_delta(e(box),e(after),300)}

class GroupTests(unittest.TestCase):
    def evaluate(self,records,groups,mode='union_support',gt=None):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'gt.json';path.write_text(json.dumps({'bboxes':[gt or {'x':0,'y':0,'w':120,'h':40,'selector':'container','type':['geometry']}]}), encoding='utf-8')
            return pipeline.evaluate_grouped(records,path,'edges8',groups,mode)

    def test_union_does_not_double_count_overlap(self):
        self.assertEqual(union_area([[0,0,10,10],[5,0,15,10]]),150)
        self.assertEqual(union_area([[0,0,10,10]]*5),100)

    def test_multiple_small_members_cover_gt(self):
        records=[record(i,[x,10,x+20,30]) for i,x in enumerate([10,40,70])]
        groups=build_prediction_groups(records,records,(300,200))
        ev=self.evaluate(records,groups)
        self.assertEqual(ev['single_covered'],0);self.assertEqual(ev['covered'],1)
        self.assertTrue(ev['group_links']);self.assertEqual(ev['group_rescued_gt'],[0])

    def test_group_coverage_has_assignments_and_member_provenance(self):
        records=[record(i,[x,10,x+20,30]) for i,x in enumerate([10,40,70])]
        groups=build_prediction_groups(records,records,(300,200))
        ev=self.evaluate(records,groups)
        self.assertEqual(ev['covered'],len(ev['assignments']))
        self.assertTrue(ev['gt_coverage'][0]['group_ids'])
        self.assertEqual(ev['gt_coverage'][0]['prediction_indices'],[0,1,2])

    def test_groups_cannot_reference_different_predictions(self):
        records=[record(i,[x,10,x+20,30]) for i,x in enumerate([10,40,70])]
        groups=build_prediction_groups(records,records,(300,200))
        groups['groups'][0]['members'][0]['before_id']='wrong'
        with self.assertRaises(ValueError):self.evaluate(records,groups)

    def test_duplicate_predictions_do_not_make_group(self):
        records=[record(0,[10,10,30,30]),record(1,[10,10,30,30])]
        self.assertFalse(build_prediction_groups(records,records,(300,200))['groups'])

    def test_inconsistent_motion_not_grouped_without_parent(self):
        records=[record(0,[10,10,30,30]),record(1,[40,10,60,30],dx=-15)]
        self.assertFalse(build_prediction_groups(records,records,(300,200))['groups'])

    def test_shared_parent_can_group_internal_reflow(self):
        records=[record(0,[10,10,30,30]),record(1,[40,10,60,30],dx=-15)]
        parent=record(2,[0,0,100,60],dx=0);parent['before_source']=parent['after_source']='medium'
        groups=build_prediction_groups(records,records+[parent],(300,200))
        self.assertTrue(any(g['kind']=='shared_parent' for g in groups['groups']))

    def test_proxy_is_separate_from_actual_union_coverage(self):
        records=[record(i,[x,x,x+10,x+15]) for i,x in enumerate([5,40,75])]
        groups=build_prediction_groups(records,records,(200,200))
        gt={'x':0,'y':0,'w':100,'h':100,'type':['geometry']}
        strict=self.evaluate(records,groups,gt=gt)
        proxy=self.evaluate(records,groups,'structural_proxy',gt)
        self.assertEqual(strict['covered'],0);self.assertEqual(proxy['covered'],1)
        self.assertTrue(any(l['evidence_type']=='structural_proxy' for l in proxy['group_links']))

    def test_grouping_never_reads_files_or_gt(self):
        records=[record(i,[x,10,x+20,30]) for i,x in enumerate([10,40,70])]
        with patch('builtins.open',side_effect=AssertionError('File read during grouping')):
            groups=build_prediction_groups(records,records,(300,200))
        self.assertFalse(groups['uses_groundtruth'])

    def test_whole_viewport_group_rejected(self):
        records=[record(0,[0,0,20,20]),record(1,[180,180,200,200],dx=-15)]
        parent=record(2,[0,0,200,200],dx=0);parent['before_source']=parent['after_source']='medium'
        self.assertFalse(build_prediction_groups(records,records+[parent],(200,200))['groups'])

if __name__=='__main__':unittest.main()
