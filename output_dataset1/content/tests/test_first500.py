import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from content.scripts import evaluate_first500 as audit


class First500Tests(unittest.TestCase):
    def test_preserves_multilabel_gt_and_original_index(self):
        targets = audit.normalize_gt({'bboxes': [
            {'type': ['style']},
            {'type': ['geometry', 'content'], 'x': 1, 'y': 2, 'w': 3, 'h': 4, 'fallback': True}]})
        self.assertEqual(targets[0]['original_gt_index'], 1)
        self.assertEqual(targets[0]['xyxy'], [1, 2, 4, 6])
        self.assertTrue(targets[0]['original']['fallback'])
        with self.assertRaises(ValueError):
            audit.normalize_gt({'before_bboxes': [], 'after_bboxes': []})

    def test_missing_id_is_blocking_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(audit, 'ROOT', root), patch.object(audit, 'REPO', root):
                out = root/'content/results/missing'
                self.assertEqual(audit.main(['--output-root', str(out), '--count', '2']), 2)
                status = json.loads((out/'preflight.json').read_text())
                self.assertEqual(status['samples_with_errors'], 2)
                self.assertFalse((out/'summary.json').exists())

    def test_two_phase_integration_and_negative_samples_not_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid in ['000', '001']:
                for side in ['baseline', 'inject']:
                    path = root/f'output_dataset1/{side}/json/{pid}_sample_full_hierarchy.json'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({'elements': [{'id': side, 'source': 'ocr',
                        'class_label': 'same', 'bbox': [10, 10, 90, 30], 'confidence': .99}]}))
                folder = root/f'dataset1/{pid}_sample'
                folder.mkdir(parents=True)
                for name in ['baseline.png', 'inject.png']:
                    Image.new('RGB', (100, 50), 'white').save(folder/name)
                gt = root/f'output_dataset1/groundtruth/{pid}.json'
                gt.parent.mkdir(parents=True, exist_ok=True)
                gt.write_text(json.dumps({'bboxes': [] if pid == '001' else [
                    {'type': ['content'], 'x': 10, 'y': 10, 'w': 80, 'h': 20}]}))
            out = root/'content/results/integration'
            original = audit.c.evaluate
            def guarded(result, gt):
                manifest = json.loads((out/'prediction_manifest.json').read_text())
                self.assertEqual(len(manifest['artifacts']), 2)
                return original(result, gt)
            with patch.object(audit, 'ROOT', root), patch.object(audit, 'REPO', root), patch.object(audit.c, 'evaluate', guarded):
                self.assertEqual(audit.main(['--output-root', str(out), '--count', '2', '--execute']), 0)
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['samples'], 2)
            self.assertEqual(summary['content_positive_samples'], 1)
            self.assertEqual(summary['pass_positive_samples'], 0)
            self.assertEqual(summary['gt_count'], 1)
            self.assertTrue(json.loads((out/'validation.json').read_text())['prediction_hashes_verified'])
            review_path = root/'review.json'
            review_path.write_text(json.dumps({'000': {'0': {
                'content_type': 'text', 'content_replaced': True, 'same_element': True,
                'presentation_unchanged': True, 'unintentional_confirmed': True,
                'evidence': 'Synthetic GT eligibility fixture, detector text intentionally unchanged'}}}))
            replacement_out = root/'content/results/replacement'
            with patch.object(audit, 'ROOT', root), patch.object(audit, 'REPO', root):
                self.assertEqual(audit.main(['--output-root', str(replacement_out), '--count', '2',
                    '--replacement', '--gt-review', str(review_path), '--execute']), 0)
            replacement_summary = json.loads((replacement_out/'summary.json').read_text())
            self.assertEqual(replacement_summary['reviewed_pure_gt'], 1)
            self.assertEqual(replacement_summary['by_reviewed_content_type']['text']['candidate_covered'], 0)
            self.assertFalse(replacement_summary['semantic_correctness_confirmed'])
            self.assertTrue((replacement_out/'candidate_review.csv').exists())
            self.assertTrue((replacement_out/'REPORT.md').exists())


if __name__ == '__main__':
    unittest.main()
