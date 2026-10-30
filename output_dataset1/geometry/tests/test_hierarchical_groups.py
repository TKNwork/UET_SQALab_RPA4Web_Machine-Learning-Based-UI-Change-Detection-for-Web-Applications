import unittest
from unittest.mock import patch
from geometry.hierarchical_groups import build_hierarchical
from geometry.prediction_groups import GROUP_CONFIG

def child(i,x,y,dx=20):
    return dict(before_id=f'b{i}',after_id=f'a{i}',before_bbox=[x,y,x+30,y+20],after_bbox=[x+dx,y,x+dx+30,y+20],delta={'dx':dx,'dy':0},before_source='ocr',after_source='ocr',match_type='ocr_anchor',before_class=f'text{i}',after_class=f'text{i}')
def parent(i,box):return dict(before_id=f'p{i}',after_id=f'q{i}',before_bbox=box,after_bbox=box,before_source='cv',after_source='cv')
def base(n):return dict(config=GROUP_CONFIG,uses_groundtruth=False,selected_predictions=n,groups=[])
def pixels(_region):return dict(change_fraction=.1,visible_fraction=1)
class HierarchyTests(unittest.TestCase):
    def test_no_file_or_gt_access_and_explicit_inferred_type(self):
        selected=[child(0,100,100),child(1,180,100),child(2,260,100)]
        with patch('builtins.open',side_effect=AssertionError('No files allowed')):
            result=build_hierarchical(selected,selected,(1000,1000),base(3),pixels)
        self.assertTrue(result['layout_proposals']);self.assertFalse(result['uses_groundtruth'])
        self.assertTrue(all(p['inferred_region_not_union_support'] for p in result['layout_proposals']))
    def test_pixel_guard_rejects_static_image(self):
        selected=[child(0,100,100),child(1,180,100),child(2,260,100)]
        result=build_hierarchical(selected,selected,(1000,1000),base(3),lambda _:dict(change_fraction=0,visible_fraction=1))
        self.assertEqual(result['layout_proposals'],[])
    def test_duplicate_detections_cannot_inflate_member_count(self):
        selected=[child(i,100,100) for i in range(3)]
        result=build_hierarchical(selected,selected,(1000,1000),base(3),pixels)
        self.assertEqual(result['layout_proposals'],[]);self.assertEqual(result['groups'],[])
    def test_inconsistent_motion_does_not_form_single_extent(self):
        selected=[child(0,100,100,20),child(1,200,100,-50),child(2,300,100,80)]
        result=build_hierarchical(selected,selected,(1000,1000),base(3),pixels)
        self.assertEqual(result['layout_proposals'],[])
    def test_cross_smallest_parent_groups_retain_common_ancestor(self):
        selected=[child(0,100,100),child(1,300,100),child(2,100,300),child(3,300,300)]
        ps=[parent(0,[80,80,370,350]),parent(1,[90,90,160,140]),parent(2,[290,290,360,340])]
        result=build_hierarchical(selected,selected+ps,(1000,1000),base(4),pixels)
        self.assertTrue(any(g['kind']=='ancestor_motion' and len(g['members'])==4 for g in result['groups']))
    def test_no_ocr_identity_support_no_layout_inference(self):
        selected=[child(0,100,100),child(1,180,100),child(2,260,100)]
        for m in selected:m['match_type']='evidence_assignment'
        result=build_hierarchical(selected,selected,(1000,1000),base(3),pixels)
        self.assertEqual(result['layout_proposals'],[])
    def test_baseline_is_retained(self):
        original={'group_id':'original','member_prediction_indices':[0,1]}
        selected=[child(0,100,100),child(1,200,100)]
        bb=base(2);bb['groups']=[dict(original,kind='baseline',parent=None)]
        result=build_hierarchical(selected,selected,(1000,1000),bb,pixels)
        self.assertEqual(result['groups'][0],bb['groups'][0])
if __name__=='__main__':unittest.main()
