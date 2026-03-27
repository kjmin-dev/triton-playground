from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.model_catalog import get_profile_model_ids, list_model_specs


class ModelCatalogTest(unittest.TestCase):
    def test_baseline_profile_only_contains_auto_download_models(self) -> None:
        baseline_ids = get_profile_model_ids("baseline")
        specs = {spec.model_id: spec for spec in list_model_specs()}

        self.assertEqual(baseline_ids, ("silero_vad",))
        for model_id in baseline_ids:
            self.assertTrue(specs[model_id].approved_for_auto_download)

    def test_hold_models_are_not_auto_downloaded(self) -> None:
        specs = {spec.model_id: spec for spec in list_model_specs()}
        self.assertFalse(specs["bs_roformer"].approved_for_auto_download)
        self.assertEqual(specs["bs_roformer"].serve_status, "hold")


if __name__ == "__main__":
    unittest.main()
