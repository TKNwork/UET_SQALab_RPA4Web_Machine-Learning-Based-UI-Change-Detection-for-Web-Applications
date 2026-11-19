import copy
import unittest

from content import pipeline as c
from content.replacement_changes import audit_gt, evaluate_reviewed, select_replacements
from content.scripts.evaluate_first500 import normalize_gt


def fixture(before='Old', after='New', label='text', shift=0):
    pair = {'before_id': 'b', 'after_id': 'a', 'before_members': ['b'], 'after_members': ['a'],
            'before_bbox': [10, 10, 110, 40], 'after_bbox': [10+shift, 10, 110+shift, 40],
            'delta': {'dx': shift, 'dy': 0}, 'content': {'state': 'unchanged', 'kind': 'none',
            'reason': 'legacy', 'before_text': before.lower(), 'after_text': after.lower(),
            'visual': {'changed_fraction': .3, 'mae': 25}, 'quality_flags': []}}
    def detection(i, text):
        return {'id': i, 'source': 'ocr' if text else 'yolo', 'class_label': text or label,
                'class_name': label, 'bbox': [10, 10, 110, 40]}
    return {'matched_pairs': [pair], 'uncertain_pairs': [], 'unmatched_before': [], 'unmatched_after': [],
            'before_detections': [detection('b', before)], 'after_detections': [detection('a', after)],
            'config': dict(c.CONFIG), 'predictions': [], 'groups': [], 'uses_groundtruth': False}


class ReplacementTests(unittest.TestCase):
    def test_preserves_raw_case_and_does_not_mutate_baseline(self):
        original = fixture('Buy', 'BUY')
        snapshot = copy.deepcopy(original)
        result = select_replacements(original)
        self.assertEqual(len(result['predictions']), 1)
        self.assertEqual(result['predictions'][0]['content']['after_text'], 'BUY')
        self.assertEqual(original, snapshot)
        self.assertIsNone(result['predictions'][0]['replacement']['semantic_confirmed'])

    def test_geometry_and_style_are_not_automatically_pure(self):
        self.assertEqual(select_replacements(fixture(shift=20))['predictions'], [])
        self.assertEqual(select_replacements(fixture('Same', 'Same'))['predictions'], [])
        self.assertEqual(select_replacements(fixture('Old', ''))['predictions'], [])
        result = select_replacements(fixture('', '', 'icon'))
        self.assertEqual(result['predictions'][0]['replacement']['content_type_hint'], 'icon')
        self.assertEqual(result['predictions'][0]['replacement']['presentation_status'], 'requires_font_color_and_style_review')
        self.assertEqual(select_replacements(fixture('', '', 'image'))['predictions'][0]['replacement']['content_type_hint'], 'image')
        self.assertEqual(select_replacements(fixture('', '', 'ui_element'))['predictions'][0]['replacement']['content_type_hint'], 'unknown')

    def test_reviewed_score_requires_evidence_and_keeps_original_indices(self):
        data = {'bboxes': [{'type': ['style']}, {'type': ['content'], 'x': 10, 'y': 10, 'w': 100, 'h': 30}]}
        targets = normalize_gt(data)
        empty = audit_gt(targets, {})
        result = select_replacements(fixture())
        self.assertIsNone(evaluate_reviewed(result, data, empty)['coverage'])
        review = {'content_type': 'text', 'content_replaced': True, 'same_element': True,
                  'presentation_unchanged': True, 'unintentional_confirmed': True, 'evidence': 'Reviewed screenshots and injection log'}
        audited = audit_gt(targets, {'1': review})
        ev = evaluate_reviewed(result, data, audited)
        self.assertEqual(ev['covered'], 1)
        self.assertEqual(ev['gt_coverage'][0]['original_gt_index'], 1)
        self.assertFalse(ev['semantic_correctness_confirmed'])
        self.assertEqual(data['bboxes'][1]['type'], ['content'])
        self.assertEqual(audit_gt(targets, {'1': {**review, 'presentation_unchanged': False}})[0]['eligibility'], 'mixed')
        self.assertEqual(audit_gt(targets, {'1': {**review, 'content_replaced': False}})[0]['eligibility'], 'excluded')
        with self.assertRaises(ValueError):
            audit_gt(targets, {'1': {**review, 'evidence': ''}})
        with self.assertRaises(ValueError):
            audit_gt(targets, {'1': {**review, 'same_element': 'true'}})
        with self.assertRaises(ValueError):
            audit_gt(targets, {'99': review})


if __name__ == '__main__':
    unittest.main()
