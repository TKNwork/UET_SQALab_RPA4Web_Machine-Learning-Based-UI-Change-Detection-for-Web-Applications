import unittest
from content import pipeline as c
from content.text_changes import normalize_text, select_text_changes


def fixture(before, after, fraction=.2):
    pair = {'before_bbox': [0, 0, 100, 20], 'after_bbox': [0, 0, 100, 20],
            'delta': {'dx': 0, 'dy': 0}, 'content': {
                'before_text': before, 'after_text': after, 'state': 'changed',
                'kind': 'visual_content_candidate', 'reason': 'legacy',
                'visual': {'changed_fraction': fraction}, 'quality_flags': []}}
    return {'config': dict(c.CONFIG), 'matched_pairs': [pair], 'uncertain_pairs': [],
            'predictions': [pair], 'groups': [], 'uses_groundtruth': False}


class TextChangesTests(unittest.TestCase):
    def test_visible_characters_are_preserved(self):
        for before, after in [('Pay 10', 'Pay 100'), ('Hello', 'hello'), ('Buy!', 'Buy?'), ('ma', 'má')]:
            result = select_text_changes(fixture(before, after))
            self.assertEqual(len(result['predictions']), 1)
        self.assertEqual(normalize_text('a\n b'), 'a b')
        self.assertEqual(normalize_text('a\u0301'), 'á')

    def test_visual_only_and_missing_ocr_are_not_text_predictions(self):
        for before, after in [('', ''), ('Old', ''), ('', 'New'), ('same', 'same'), ('a\n b', 'a b')]:
            source = fixture(before, after)
            self.assertEqual(select_text_changes(source)['predictions'], [])
            self.assertEqual(source['matched_pairs'][0]['content']['state'], 'changed')

    def test_ocr_difference_without_pixels_is_uncertain(self):
        result = select_text_changes(fixture('Old', 'New', 0))
        self.assertEqual(result['predictions'], [])
        self.assertEqual(result['matched_pairs'][0]['content']['state'], 'uncertain')


if __name__ == '__main__':
    unittest.main()
