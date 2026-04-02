from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comic_pipeline.step1 import summarize_step1_validation
from comic_pipeline.types import BalloonPrediction


class Step1ValidationTests(unittest.TestCase):
    def test_validation_passes_for_reasonable_mask(self) -> None:
        predictions = [
            BalloonPrediction(
                bbox_xyxy=[10, 10, 110, 110],
                polygon=[[10, 10], [110, 10], [110, 110], [10, 110]],
                area=10000.0,
                confidence=0.9,
                model_name="manga109_seg_v1",
            )
        ]
        report = summarize_step1_validation(
            page_id="0001",
            width=1000,
            height=1000,
            predictions=predictions,
            mask_nonzero_pixels=15000,
        )
        self.assertTrue(report.passed)
        self.assertFalse(report.bbox_out_of_bounds)
        self.assertTrue(report.production_detector_active)

    def test_validation_fails_for_empty_mask(self) -> None:
        report = summarize_step1_validation(
            page_id="0002",
            width=1000,
            height=1000,
            predictions=[],
            mask_nonzero_pixels=0,
        )
        self.assertFalse(report.passed)
        self.assertIn("mask is empty", report.notes[0])

    def test_validation_fails_for_baseline_only_detector(self) -> None:
        predictions = [
            BalloonPrediction(
                bbox_xyxy=[10, 10, 110, 110],
                polygon=[[10, 10], [110, 10], [110, 110], [10, 110]],
                area=10000.0,
                confidence=0.9,
                model_name="contour_baseline_v1",
            )
        ]
        report = summarize_step1_validation(
            page_id="0003",
            width=1000,
            height=1000,
            predictions=predictions,
            mask_nonzero_pixels=15000,
        )
        self.assertFalse(report.passed)
        self.assertFalse(report.production_detector_active)
        self.assertTrue(any("baseline contour detector" in note for note in report.notes))


if __name__ == "__main__":
    unittest.main()
