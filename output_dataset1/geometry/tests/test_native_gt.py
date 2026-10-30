import unittest
from geometry.native_gt_evaluation import evaluate_native

def record(b,a):
    return dict(before_id='b',after_id='a',before_bbox=b,after_bbox=a,delta=dict(center_shift=20,max_edge_shift=20,relative_dw=0,relative_dh=0))
def gtbox(x,selector='.same'):
    return dict(x=x,y=0,w=10,h=10,selector=selector,type=['geometry'])
def groups(records):
    return dict(uses_groundtruth=False,selected_predictions=len(records),groups=[])
class NativeGTTests(unittest.TestCase):
    def test_repeated_selectors_different_counts_all_scored(self):
        rs=[record([0,0,10,10],[20,0,30,10])]
        ev=evaluate_native(rs,dict(before_bboxes=[gtbox(0),gtbox(50)],after_bboxes=[gtbox(20)]),groups(rs))
        self.assertEqual(ev['gt_count'],3);self.assertEqual(ev['covered'],2);self.assertFalse(ev['pass'])
    def test_envelope_representation_preserved_across_gt_sides(self):
        rs=[record([0,0,10,10],[20,0,30,10])]
        ev=evaluate_native(rs,dict(before_bboxes=[gtbox(20)],after_bboxes=[]),groups(rs))
        self.assertEqual(ev['covered'],1)
    def test_single_side_only_is_scored(self):
        rs=[record([0,0,10,10],[20,0,30,10])]
        ev=evaluate_native(rs,dict(before_bboxes=[],after_bboxes=[gtbox(20)]),groups(rs))
        self.assertTrue(ev['pass']);self.assertEqual(ev['gt_count'],1)
    def test_empty_geometry_raises(self):
        with self.assertRaises(ValueError):evaluate_native([],dict(before_bboxes=[],after_bboxes=[]),groups([]))
    def test_motion_envelope_gap_matches_legacy_and_is_flagged(self):
        rs=[record([0,0,10,10],[100,0,110,10])]
        ev=evaluate_native(rs,dict(before_bboxes=[gtbox(50)],after_bboxes=[gtbox(50)]),groups(rs))
        self.assertEqual(ev['covered'],2)
        self.assertTrue(ev['gt_coverage'][0]['links'][0]['envelope_gap_only_on_gt_side'])
    def test_group_exact_union_not_enclosing_box(self):
        rs=[record([0,0,10,10],[0,20,10,30]),record([90,0,100,10],[90,20,100,30])]
        rs[1]['before_id']='b2';rs[1]['after_id']='a2'
        members=[dict(prediction_index=i,**{k:r[k] for k in ('before_id','after_id','before_bbox','after_bbox')}) for i,r in enumerate(rs)]
        gs=dict(uses_groundtruth=False,selected_predictions=2,groups=[dict(group_id='g',members=members,member_prediction_indices=[0,1])])
        ev=evaluate_native(rs,dict(before_bboxes=[dict(x=0,y=0,w=100,h=10,type=['geometry'])],after_bboxes=[]),gs)
        self.assertEqual(ev['covered'],0)
    def test_union_can_rescue_multiple_small_predictions(self):
        rs=[record([0,0,25,10],[0,20,25,30]),record([25,0,50,10],[25,20,50,30])]
        rs[1]['before_id']='b2';rs[1]['after_id']='a2'
        members=[dict(prediction_index=i,**{k:r[k] for k in ('before_id','after_id','before_bbox','after_bbox')}) for i,r in enumerate(rs)]
        gs=dict(uses_groundtruth=False,selected_predictions=2,groups=[dict(group_id='g',members=members,member_prediction_indices=[0,1])])
        data=dict(before_bboxes=[dict(x=0,y=0,w=100,h=10,type=['geometry'])],after_bboxes=[])
        self.assertEqual(evaluate_native(rs,data,gs,'edges8','single_envelope')['covered'],0)
        self.assertEqual(evaluate_native(rs,data,gs)['covered'],1)
    def test_duplicate_gt_annotations_preserve_original_indices(self):
        rs=[record([0,0,10,10],[20,0,30,10])]
        ev=evaluate_native(rs,dict(before_bboxes=[gtbox(0),gtbox(0)],after_bboxes=[]),groups(rs))
        self.assertEqual(ev['gt_count'],2);self.assertEqual([r['original_gt_index'] for r in ev['gt_coverage']],[0,1])
if __name__=='__main__':unittest.main()
