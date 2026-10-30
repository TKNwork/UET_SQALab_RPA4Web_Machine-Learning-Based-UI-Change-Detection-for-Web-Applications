import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import numpy as np
from geometry import pipeline as p


def elem(i, text='', source='medium', bbox=None):
    return p.normalize_detection({'id':i,'source':'ocr' if text else source,'class_label':text or 'ui_element','bbox':bbox or [0,0,100,100],'confidence':.99},i)


class PipelineTests(unittest.TestCase):
    def test_text_container_not_merged(self):
        es=[elem('text','hello'),elem('box')]
        info=p.build_duplicate_groups(es,p.DEFAULT_CONFIG)
        reps=p.group_representative_elems(es,info)
        self.assertEqual(len(reps),2)
        self.assertEqual(sum(e['has_ocr'] for e in reps),1)

    def test_duplicate_provenance_preserved(self):
        es=[elem('a','hello'),elem('b','hello',bbox=[1,1,99,99])]
        reps=p.group_representative_elems(es,p.build_duplicate_groups(es,p.DEFAULT_CONFIG))
        self.assertEqual(len(reps),1)
        self.assertEqual(len(reps[0]['ocr_members']),2)
        self.assertEqual(set(reps[0]['original_bboxes']),{'a','b'})

    def test_near_exact_not_locked(self):
        r=p.build_ocr_anchors([elem('b','a long button label')],[elem('a','a long button labe1')],p.DEFAULT_CONFIG)
        self.assertFalse(r['locked']);self.assertTrue(r['candidates'])

    def test_local_prior_ignores_far_anchor(self):
        anchors={'local_samples':[{'x':50,'y':50,'dx':0,'dy':0},{'x':950,'y':950,'dx':500,'dy':500}]}
        detail=p.build_evidence_cost(elem('b'),elem('a'),1500,anchors,p.DEFAULT_CONFIG,return_detail=True)
        self.assertEqual(detail['local_anchor_count'],1)
        self.assertEqual(detail['components']['shift_prior'],0)

    def test_stability_uses_total_assignment_cost(self):
        costs=np.array([[.1,.11],[.1,1.]])
        bs=[elem('b0'),elem('b1')];as_=[elem('a0'),elem('a1')]
        def fake(b,a,*args,**kwargs):
            c=float(costs[int(b['id'][1:]),int(a['id'][1:])]);return {'cost':c} if kwargs.get('return_detail') else c
        with patch.object(p,'build_evidence_cost',side_effect=fake):
            r=p.solve_assignment(bs,as_,set(),set(),100,{},p.DEFAULT_CONFIG)
        self.assertEqual(len(r['matched_pairs']),2)
        self.assertFalse(r['uncertain_pairs'])
        for m in r['matched_pairs']:self.assertAlmostEqual(m['evidence']['cost_gap_ban'],.89)

    def test_tied_assignment_uncertain(self):
        def fake(*args,**kwargs):return {'cost':.1} if kwargs.get('return_detail') else .1
        with patch.object(p,'build_evidence_cost',side_effect=fake):
            r=p.solve_assignment([elem('b0'),elem('b1')],[elem('a0'),elem('a1')],set(),set(),100,{},p.DEFAULT_CONFIG)
        self.assertEqual(len(r['uncertain_pairs']),2);self.assertFalse(r['matched_pairs'])

    def test_infeasible_stays_unmatched(self):
        r=p.solve_assignment([elem('b','hello')],[elem('a','xxxxxxxx')],set(),set(),100,{},p.DEFAULT_CONFIG)
        self.assertEqual(len(r['unmatched_before']),1);self.assertEqual(len(r['unmatched_after']),1)

    def test_symmetric_resize_detected_by_edges(self):
        b=elem('b',bbox=[100,100,700,400]);a=elem('a',bbox=[70,85,730,415]);d=p.calc_delta(b,a,1000)
        self.assertFalse(p.classify_change(d,'relative15'));self.assertTrue(p.classify_change(d,'edges8'))

    def test_no_gt_needed_for_matching(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pair=root/'000';pair.mkdir()
            for side in ('before','after'):(pair/f'{side}.json').write_text(json.dumps({'elements':[{'id':side,'source':'ocr','class_label':'hello','bbox':[10,10,100,30]}]}), encoding='utf-8')
            r=p.process_pair(pair,root/'out',p.DEFAULT_CONFIG)
            self.assertEqual(len(r['matched_pairs']),1)

    def test_cli_rejects_output_alias(self):
        import subprocess,sys,json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pair=root/'000';pair.mkdir()
            for side in ('before','after'):(pair/f'{side}.json').write_text(json.dumps({'elements':[]}), encoding='utf-8')
            run=subprocess.run([sys.executable,'-m','geometry.match_geometry','--input-root',str(root),'--output-root',str(root),'--pair-ids','000'],capture_output=True)
            self.assertNotEqual(run.returncode,0)
            self.assertFalse((root/'config.json').exists())

    def test_cli_requires_explicit_pair_ids(self):
        import subprocess,sys
        run=subprocess.run([sys.executable,'-m','geometry.match_geometry'],capture_output=True)
        self.assertNotEqual(run.returncode,0)

    def test_duplicate_ids_fail_instead_of_silent_accounting(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.json'
            path.write_text(json.dumps({'elements':[{'id':'same','bbox':[0,0,10,10]}]*2}), encoding='utf-8')
            with self.assertRaises(ValueError):p.load_normalized(path)

    def test_all_geometry_selects_only_geometry_and_completes(self):
        import subprocess,sys,json
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'input';source.mkdir()
            for pid,kind in [('000','geometry'),('001','style')]:
                pair=source/pid;pair.mkdir()
                for side in ('before','after'):
                    (pair/f'{side}.json').write_text(json.dumps({'elements':[]}), encoding='utf-8')
                    Image.new('RGB',(8,8)).save(pair/f'{side}.png')
                (pair/'groundtruth.json').write_text(json.dumps({'bboxes':[{'x':1,'y':1,'w':2,'h':2,'type':[kind]}]}), encoding='utf-8')
            out=root/'out'
            run=subprocess.run([sys.executable,'-m','geometry.match_geometry','--input-root',str(source),'--output-root',str(out),'--all-geometry'],capture_output=True)
            self.assertEqual(run.returncode,0,run.stderr.decode())
            manifest=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['pair_ids'],['000'])
            self.assertTrue(manifest['input_hashes_verified'])
            self.assertEqual(json.loads((out/'run_status.json').read_text(encoding='utf-8'))['status'],'complete')
            self.assertFalse((out/'001').exists())

class EnvelopeEvaluationTests(unittest.TestCase):
    def evaluate(self, boxes, records, protocol=p.EVALUATION_PROTOCOL):
        import json
        with tempfile.TemporaryDirectory() as folder:
            gt=Path(folder)/'groundtruth.json';gt.write_text(json.dumps({'bboxes':boxes}), encoding='utf-8')
            return p.evaluate(records,gt,'edges8',protocol=protocol)

    def record(self, before, after):
        b,a=elem('before',bbox=before),elem('after',bbox=after)
        return {'before_id':'before','after_id':'after','before_bbox':before,'after_bbox':after,'delta':p.calc_delta(b,a,1000)}

    def gt(self,x,y,w,h,kind='geometry'):
        return {'x':x,'y':y,'w':w,'h':h,'type':[kind],'selector':'button','fallback':False}

    def test_prediction_region_covers_15px_translation(self):
        record=self.record([100,100,200,140],[115,100,215,140])
        ev=self.evaluate([self.gt(100,100,115,40)],[record])
        self.assertEqual(ev['prediction_regions'][0]['prediction_region'],[100,100,215,140])
        self.assertTrue(ev['pass']);self.assertAlmostEqual(ev['assignments'][0]['iou'],1)

    def test_one_prediction_can_cover_two_gt_unverified_not_negative(self):
        record=self.record([0,0,100,40],[150,0,250,40])
        gt=[self.gt(0,0,100,40),self.gt(150,0,100,40)]
        ev=self.evaluate(gt,[record])
        self.assertEqual(ev['covered'],2)
        self.assertEqual(ev['predictions'],1);self.assertEqual(ev['unverified'],0)
        self.assertEqual(ev['supporting_predictions'],1)

    def test_gap_only_match_is_explicitly_flagged(self):
        record=self.record([0,0,20,20],[100,0,120,20])
        ev=self.evaluate([self.gt(45,0,20,20)],[record])
        self.assertTrue(ev['assignments'][0]['envelope_gap_only'])

    def test_unchanged_matched_pair_not_selected(self):
        record=self.record([0,0,100,40],[0,0,100,40])
        ev=self.evaluate([self.gt(0,0,100,40)],[record])
        self.assertFalse(ev['pass']);self.assertEqual(ev['predictions'],0)

    def test_original_gt_index_and_geometry_filter(self):
        record=self.record([0,0,100,40],[15,0,115,40])
        ev=self.evaluate([self.gt(0,0,100,40,'style'),self.gt(0,0,115,40)],[record])
        self.assertEqual(ev['gt_count'],1);self.assertEqual(ev['assignments'][0]['original_gt_index'],1)

    def test_prediction_without_gt_is_unverified_not_failure(self):
        record=self.record([0,0,100,40],[15,0,115,40])
        ev=self.evaluate([],[record]);self.assertTrue(ev['pass']);self.assertEqual(ev['unverified'],1)

    def test_exact_half_gt_coverage_accepted(self):
        record=self.record([-200,0,-100,100],[-50,0,50,100])
        ev=self.evaluate([self.gt(0,0,100,100)],[record])
        self.assertTrue(ev['pass'])

    def test_invalid_envelope_rejected(self):
        with self.assertRaises(ValueError):p.prediction_envelope([1,1,0,2],[1,1,2,2])

if __name__=='__main__':unittest.main()
