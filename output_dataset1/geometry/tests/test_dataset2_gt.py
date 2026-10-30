import unittest
from geometry.scripts.normalize_dataset2_gt import normalize

def b(x,selector='item'):
    return dict(x=x,y=10,w=20,h=10,selector=selector,type=['geometry'])
def dom(x,selector):
    return dict(bbox=dict(x=x+.7,y=10.,width=20.,height=10.),selector=selector)
class PairedGTTests(unittest.TestCase):
    def test_unique_selector_encloses_both_endpoints(self):
        r=normalize(dict(pair_id='502',before_bboxes=[b(10)],after_bboxes=[b(25)]),{'before':[],'after':[]})
        self.assertTrue(r['fully_resolved']);self.assertEqual(r['bboxes'][0]['w'],35)
    def test_repeated_selector_without_identity_is_unresolved(self):
        r=normalize(dict(pair_id='502',before_bboxes=[b(10),b(40)],after_bboxes=[b(25),b(55)]),{'before':[],'after':[]})
        self.assertFalse(r['fully_resolved']);self.assertEqual(r['bboxes'],[])
    def test_unique_dom_identity_handles_reversed_gt_order(self):
        r=normalize(dict(pair_id='502',before_bboxes=[b(10),b(40)],after_bboxes=[b(55),b(25)]),{'before':[dom(10,'#a'),dom(40,'#b')],'after':[dom(25,'#a'),dom(55,'#b')]})
        self.assertTrue(r['fully_resolved']);self.assertEqual([x['w'] for x in r['bboxes']],[35,35])
        self.assertEqual(r['bboxes'][0]['provenance']['after_index'],1)
    def test_missing_endpoint_cannot_be_full_pass(self):
        r=normalize(dict(pair_id='502',before_bboxes=[b(10)],after_bboxes=[]),{'before':[],'after':[]})
        self.assertFalse(r['fully_resolved']);self.assertEqual(r['bboxes'],[])
if __name__=='__main__':unittest.main()
