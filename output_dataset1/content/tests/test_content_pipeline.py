import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image,ImageDraw
from content import pipeline as c
from geometry import pipeline as p


def elem(i,text='',box=None):
    return p.normalize_detection({'id':i,'source':'ocr' if text else 'medium','class_label':text or 'ui_element','bbox':box or [10,10,150,40],'confidence':.99},i)


def picture(text='Alpha',color='white'):
    im=Image.new('RGB',(240,100),color);ImageDraw.Draw(im).text((20,20),text,fill='black');return im


class ContentTests(unittest.TestCase):
    def test_entirely_different_ocr_is_feasible(self):
        b,a=elem('b','apply'),elem('a','complaint')
        self.assertIsNone(p.build_evidence_cost(b,a,300,{},p.DEFAULT_CONFIG))
        self.assertIsNotNone(c.content_cost(b,a,300,{},c.CONFIG))
        r=p.solve_assignment([b],[a],set(),set(),300,{},c.CONFIG,cost_builder=c.content_cost)
        self.assertEqual(len(r['matched_pairs']),1)

    def test_missing_ocr_can_correspond_to_visual_detection(self):
        d=c.content_cost(elem('b'),elem('a','new label'),300,{},c.CONFIG,return_detail=True)
        self.assertGreater(d['components']['level_transition'],0)

    def test_repeated_text_uses_position_and_stays_one_to_one(self):
        bs=[elem('b1','same',[10,10,100,30]),elem('b2','same',[10,60,100,80])]
        ats=[elem('a1','same',[10,10,100,30]),elem('a2','same',[10,60,100,80])]
        r=p.solve_assignment(bs,ats,set(),set(),300,{},c.CONFIG,cost_builder=c.content_cost)
        self.assertEqual({(m['before_id'],m['after_id']) for m in r['matched_pairs']},{('b1','a1'),('b2','a2')})

    def test_same_text_ocr_noise_and_split_merge(self):
        b,a=elem('b','hello world'),elem('a','hello\nworld')
        self.assertEqual(c.classify(b,a,picture(),picture(),[b],[a],c.CONFIG)['state'],'unchanged')
        a=elem('a','hello wor1d')
        self.assertEqual(c.classify(b,a,picture(),picture(),[b],[a],c.CONFIG)['state'],'uncertain')
        container=elem('container',box=[0,0,200,80]);raw=[elem('a','hello',[10,10,50,30]),elem('b','world',[60,10,100,30])]
        self.assertEqual(c.canonical(c.contained_text(container,raw)),c.canonical('hello world'))

    def test_unchanged_text_with_motion_and_crop_jitter(self):
        bi=picture('Find similar domains');ai=Image.new('RGB',(240,100),'white');ai.paste(bi,(0,20))
        b=elem('b',box=[18.2,19.1,142.3,29.8]);a=elem('a',box=[19.1,39.3,142.1,51.0])
        evidence=c.crop_evidence(b,a,bi,ai,c.CONFIG)
        self.assertLess(evidence['changed_fraction'],.05)

    def test_color_only_not_semantic_content(self):
        b,a=elem('b'),elem('a')
        info=c.classify(b,a,Image.new('RGB',(240,100),'red'),Image.new('RGB',(240,100),'blue'),[b],[a],c.CONFIG)
        self.assertEqual(info['state'],'unchanged')

    def test_visual_change_does_not_require_ocr(self):
        b,a=elem('b',box=[0,0,90,35]),elem('a',box=[0,0,90,35])
        bi=Image.new('RGB',(240,100),'white');ai=bi.copy();ImageDraw.Draw(ai).rectangle([10,10,70,30],fill='black')
        info=c.classify(b,a,bi,ai,[b],[a],c.CONFIG)
        self.assertEqual(info['state'],'changed');self.assertEqual(info['kind'],'visual_content_candidate')

    def test_gt_free_inference_and_duplicates_accounting(self):
        with tempfile.TemporaryDirectory() as f:
            root=Path(f)
            for side in ['before','after']:
                data={'elements':[{'id':side+str(i),'bbox':[10+i,10,150,40],'source':'ocr','class_label':'same','confidence':.99} for i in range(2)]}
                (root/(side+'.json')).write_text(json.dumps(data), encoding='utf-8');picture().save(root/(side+'.png'))
            original=Path.read_text
            def guarded(path,*args,**kwargs):
                self.assertNotIn('groundtruth',str(path));return original(path,*args,**kwargs)
            with patch.object(Path,'read_text',guarded):
                r=c.infer(root/'before.json',root/'after.json',root/'before.png',root/'after.png')
            self.assertFalse(r['uses_groundtruth']);self.assertEqual(len(r['matched_pairs']),1)
            self.assertEqual(len(r['matched_pairs'][0]['before_members']),2)
            before=copy.deepcopy(r)
            c.evaluate(r,{'bboxes':[{'x':0,'y':0,'w':100,'h':100,'type':['content']}]})
            self.assertEqual(r,before)

    def test_no_motion_gap_coverage(self):
        r={'predictions':[{'before_bbox':[0,0,10,10],'after_bbox':[90,0,100,10]}], 'groups':[]}
        ev=c.evaluate(dict(matched_pairs=[],uncertain_pairs=[],unmatched_before=[],unmatched_after=[],before_detections=[],after_detections=[],**r),{'bboxes':[{'x':40,'y':0,'w':20,'h':10,'type':['content']}]})
        self.assertEqual(ev['covered'],0)

    def test_many_gt_and_original_indices_and_unverified(self):
        r={'predictions':[{'before_bbox':[0,0,10,10],'after_bbox':[20,0,30,10]}, {'before_bbox':[100,0,110,10],'after_bbox':[100,0,110,10]}], 'groups':[]}
        ev=c.evaluate(dict(matched_pairs=[],uncertain_pairs=[],unmatched_before=[],unmatched_after=[],before_detections=[],after_detections=[],**r),{'bboxes':[{'x':0,'y':0,'w':10,'h':10,'type':['style']},{'x':0,'y':0,'w':10,'h':10,'type':['content','geometry']},{'x':20,'y':0,'w':10,'h':10,'type':['content']}]})
        self.assertEqual(ev['covered'],2);self.assertEqual(ev['unverified_predictions'],1)
        self.assertEqual([g['original_gt_index'] for g in ev['gt_coverage']],[1,2])

    def test_large_region_cannot_cover_tiny_gt_by_containment_alone(self):
        r={'predictions':[{'before_bbox':[0,0,1000,1000],'after_bbox':[0,0,1000,1000]}], 'groups':[]}
        ev=c.evaluate(dict(matched_pairs=[],uncertain_pairs=[],unmatched_before=[],unmatched_after=[],before_detections=[],after_detections=[],**r),{'bboxes':[{'x':0,'y':0,'w':10,'h':10,'type':['content']}]})
        self.assertEqual(ev['covered'],0)

    def test_group_union_no_double_count(self):
        self.assertEqual(c.support_metrics([[0,0,10,10]]*4,[0,0,20,10])['gt_fraction'],.5)
        with patch('builtins.open',side_effect=AssertionError('groups must not read GT')):
            self.assertEqual(c.groups([],c.CONFIG),[])

if __name__=='__main__':unittest.main()
