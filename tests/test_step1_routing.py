from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comic_pipeline.step1 import resolve_detector_name


class Step1RoutingTests(unittest.TestCase):
    def test_auto_resolves_to_manga109_for_all_known_profiles(self) -> None:
        for profile in ["bw_manga", "color_comic", "three_d_comic", "unknown"]:
            with self.subTest(profile=profile):
                self.assertEqual(resolve_detector_name(profile, "auto"), "manga109_seg_v1")

    def test_explicit_detector_still_respects_manual_override(self) -> None:
        self.assertEqual(resolve_detector_name("color_comic", "kitsumed_seg_v1"), "kitsumed_seg_v1")


if __name__ == "__main__":
    unittest.main()
