import unittest
from content.scripts.build_content_catalog import SavedDOM,css_match,tokens,visible_text


class CatalogSelectorTests(unittest.TestCase):
    def setUp(self):
        self.dom=SavedDOM('<main id="hero"><h1 class="title lead">Old <b>heading</b></h1><h1 class="title">Second</h1><div><input name="email" placeholder="your email"><a href="/my-first-job-interview/">Comments</a></div></main>')
    def select(self,selector):return [n for n in self.dom.nodes if css_match(n,selector)]
    def test_descendant_child_and_nth_type(self):
        self.assertEqual(visible_text(self.select('#hero > h1.title:nth-of-type(2)')[0]),'Second')
        self.assertEqual(len(self.select('main h1')),2)
        self.assertFalse(self.select('main > input'))
    def test_attribute_value_with_space_and_contains(self):
        self.assertEqual(len(self.select('input[placeholder="your email"]')),1)
        self.assertEqual(len(self.select('a[href*="my-first-job-interview"]')),1)
        self.assertEqual(len(self.select('[name="email"]')),1)
    def test_unsupported_selector_is_not_guessed(self):
        self.assertFalse(self.select('h1:madeup'))
        self.assertFalse(self.select('h1 + div'))
    def test_nested_text_order(self):
        self.assertEqual(visible_text(self.select('h1:first-child')[0]),'Old heading')

if __name__=='__main__':unittest.main()
