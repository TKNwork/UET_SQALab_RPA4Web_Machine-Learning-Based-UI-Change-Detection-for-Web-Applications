import unittest
import os
from unittest.mock import patch
from shared.paths import ROOT,resolve_recorded_path
from shared.matching import solve_assignment
from shared.paths import resource_path


class LayoutTests(unittest.TestCase):
    def test_recorded_linux_and_windows_paths_rebase_after_clone(self):
        for prefix in ['/home/previous/RPA/', 'C:\\old\\RPA\\']:
            sep = '\\' if prefix.startswith('C:') else '/'
            value = prefix + sep.join(['output_dataset1', 'geometry_results', 'missing', 'result.json'])
            self.assertEqual(resolve_recorded_path(value), ROOT/'geometry/results/missing/result.json')

    def test_relative_resource_override_uses_repository_root(self):
        with patch.dict(os.environ, {'RPA_DATASET1': 'external/screenshots'}):
            self.assertEqual(resource_path('dataset1', 'dataset1'), ROOT.parent/'external/screenshots')

    def test_legacy_manifest_paths_resolve_without_mutating_manifests(self):
        self.assertEqual(resolve_recorded_path('geometry_pairs/000/before.json'),ROOT/'geometry/pairs/000/before.json')
        self.assertEqual(resolve_recorded_path(ROOT/'geometry_results/matching_manifest.json'),ROOT/'geometry/results/matching_manifest.json')
        self.assertEqual(resolve_recorded_path('content_results/v1/config.json'),ROOT/'content/results/v1/config.json')

    def test_common_assignment_requires_explicit_task_cost(self):
        with self.assertRaises(ValueError):solve_assignment([],[],set(),set(),1,{}, {})

if __name__=='__main__':unittest.main()
