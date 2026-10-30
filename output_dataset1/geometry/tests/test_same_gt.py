import unittest
from geometry.scripts.evaluate_dataset2_same_gt import geometry_boxes

class CanonicalGTTests(unittest.TestCase):
    def test_paired_schema_is_rejected_not_zero_gt(self):
        with self.assertRaises(ValueError):geometry_boxes({'before_bboxes':[],'after_bboxes':[]})
    def test_multilabel_fallback_outside_and_duplicate_annotations_are_preserved(self):
        b={'x':-5,'y':-10,'w':20,'h':30,'selector':'.same','type':['style','geometry'],'fallback':True}
        result=geometry_boxes({'bboxes':[dict(type=['content']),b,dict(b)]})
        self.assertEqual([r['original_index'] for r in result],[1,2])
        self.assertEqual(result[0]['annotation'],b);self.assertEqual(result[1]['annotation'],b)
    def test_degenerate_geometry_raises_before_evaluator(self):
        with self.assertRaises(ValueError):geometry_boxes({'bboxes':[{'x':0,'y':0,'w':0,'h':5,'type':['geometry']}]})
if __name__=='__main__':unittest.main()
