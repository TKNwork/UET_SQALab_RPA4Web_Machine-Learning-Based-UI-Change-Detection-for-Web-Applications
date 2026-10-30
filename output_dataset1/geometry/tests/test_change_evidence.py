import unittest
from PIL import Image,ImageDraw
from geometry.change_evidence import motion_features,event_links
class EvidenceTests(unittest.TestCase):
    def test_detector_jitter_on_unchanged_image_rejected(self):
        im=Image.new('RGB',(100,100),'white');ImageDraw.Draw(im).rectangle([20,20,40,40],fill='black');r={'before_bbox':[15,15,45,45],'after_bbox':[25,15,55,45],'match_type':'ocr_anchor','before_class':'ABC','after_class':'ABC'}
        self.assertFalse(motion_features(im,im,r)['strong'])
    def test_real_translation_has_motion_evidence(self):
        b=Image.new('RGB',(120,100),'white');a=b.copy();ImageDraw.Draw(b).rectangle([20,20,35,35],fill='black');ImageDraw.Draw(a).rectangle([50,20,65,35],fill='black');r={'before_bbox':[15,15,40,40],'after_bbox':[45,15,70,40],'match_type':'ocr_anchor','before_class':'ABC','after_class':'ABC'}
        self.assertTrue(motion_features(b,a,r)['strong'])
    def test_blank_background_rejected(self):
        im=Image.new('RGB',(100,100),'white');r={'before_bbox':[0,0,20,20],'after_bbox':[20,0,40,20],'match_type':'ocr_anchor','before_class':'ABC','after_class':'ABC'}
        self.assertFalse(motion_features(im,im,r)['strong'])
    def test_coarse_gt_containment_is_not_area_recall(self):
        g={'group_id':'g','members':[{'visual_evidence':{'strong':True}}]*2,'original_prediction_indices':[0,1],'member_regions':[[0,0,10,10],[20,0,30,10]]};links=event_links([g],[0,0,100,100]);self.assertEqual(len(links),1);self.assertAlmostEqual(links[0]['gt_fraction'],.02)
        g['members'][1]={'visual_evidence':{'strong':False}};self.assertEqual(event_links([g],[0,0,100,100]),[])
    def test_gt_inside_large_group_rejected(self):
        g={'group_id':'g','members':[{'visual_evidence':{'strong':True}}]*2,'original_prediction_indices':[0,1],'member_regions':[[0,0,100,100],[100,0,200,100]]};self.assertEqual(event_links([g],[0,0,10,10]),[])
if __name__=='__main__':unittest.main()
