from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import comic_pipeline  # noqa: F401

import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from comic_pipeline.project import load_page_manifest, read_json
from comic_pipeline.step4 import restore_alpha
from comic_pipeline.step5 import compose_final, validate_step5
from comic_pipeline.step6 import validate_step6
from comic_pipeline.types import BalloonPrediction
from step3_test_utils import build_phase2_ready_page, build_phase4_ready_page


class _StaticDetector:
    def __init__(self, predictions: list[BalloonPrediction]) -> None:
        self.name = "manga109_seg_v1"
        self._predictions = predictions

    def predict(self, image: object) -> list[BalloonPrediction]:
        return list(self._predictions)


class Step6ValidationTests(unittest.TestCase):
    def test_validate_step6_single_page_reports_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase4_ready_page(project_root)
            compose_final(project_root, "0001")

            report = validate_step6(project_root, page_id="0001")

            self.assertTrue(report.passed)
            self.assertEqual(report.done_pages, ["0001"])
            self.assertEqual(report.check_pages, [])
            self.assertTrue(report.balloon_match_summary)

    def test_validate_step6_all_reports_missing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase4_ready_page(project_root, page_id="0001")
            compose_final(project_root, "0001")
            build_phase2_ready_page(project_root, page_id="0002")

            report = validate_step6(project_root, validate_all=True)

            self.assertFalse(report.passed)
            self.assertIn("0002", report.check_pages)
            self.assertIn("0002", report.missing_artifact_pages)

    def test_validate_step6_accepts_manual_step4_passthrough_without_step3_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase2_ready_page(project_root)

            manual = np.full((240, 180, 3), 245, dtype=np.uint8)
            cv2.ellipse(manual, (90, 95), (32, 20), 0, 0, 360, (180, 180, 180), -1)
            cv2.putText(
                manual,
                "KO",
                (72, 101),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )
            manual_path = project_root / "imports" / "nano" / "0001_submitted.png"
            cv2.imwrite(str(manual_path), manual)

            page_manifest = load_page_manifest(project_root, "0001")
            predictions = [
                BalloonPrediction(
                    bbox_xyxy=item["bbox_xyxy"],
                    polygon=item["polygon"],
                    area=float(item.get("area", 0.0)),
                    confidence=float(item.get("confidence", 0.0)),
                    model_name="manga109_seg_v1",
                )
                for item in page_manifest.balloons
            ]
            with mock.patch(
                "comic_pipeline.registration.build_detector",
                return_value=_StaticDetector(predictions),
            ):
                restored = restore_alpha(project_root, "0001")
            self.assertTrue(restored["passed"])
            compose_final(project_root, "0001")

            report = validate_step6(project_root, page_id="0001")

            self.assertTrue(report.passed)
            self.assertEqual(report.done_pages, ["0001"])

    def test_validate_step6_reports_low_balloon_match_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase4_ready_page(project_root)
            compose_final(project_root, "0001")

            page_manifest = load_page_manifest(project_root, "0001")
            registration_path = project_root / page_manifest.registration_report_path
            payload = read_json(registration_path)
            payload["detector_alignment_mode"] = "manga109_seg_v1"
            payload["reference_balloon_count"] = 1
            payload["matched_balloon_count"] = 1
            payload["matched_ratio"] = 0.50
            payload["median_balloon_iou"] = 0.40
            registration_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            validate_step5(project_root, "0001")

            report = validate_step6(project_root, page_id="0001")

            self.assertFalse(report.passed)
            self.assertIn("0001", report.low_balloon_match_pages)
            self.assertTrue(report.balloon_match_summary)


if __name__ == "__main__":
    unittest.main()
