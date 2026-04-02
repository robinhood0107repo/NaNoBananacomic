from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import comic_pipeline  # noqa: F401
import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from comic_pipeline.project import load_page_manifest, read_json
from comic_pipeline.step5 import compose_final, validate_step5
from step3_test_utils import build_phase4_ready_page


class Step5CompositeTests(unittest.TestCase):
    def test_compose_final_writes_outputs_and_marks_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase4_ready_page(project_root)

            result = compose_final(project_root, "0001")
            report = validate_step5(project_root, "0001")
            page_manifest = load_page_manifest(project_root, "0001")

            self.assertTrue(result["passed"])
            self.assertTrue(report.passed)
            self.assertEqual(page_manifest.status, "done")
            self.assertTrue((project_root / page_manifest.final_composite_path).exists())
            self.assertTrue((project_root / page_manifest.result_path).exists())
            self.assertTrue((project_root / page_manifest.diff_preview_path).exists())

    def test_validate_step5_marks_check_for_severe_registration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase4_ready_page(project_root, mismatch_manual_import=True)

            page_manifest = load_page_manifest(project_root, "0001")
            registration_path = project_root / page_manifest.registration_report_path
            payload = read_json(registration_path)
            payload["warning_level"] = "severe"
            payload["final_score"] = 0.31
            registration_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            result = compose_final(project_root, "0001")
            report = validate_step5(project_root, "0001")

            self.assertFalse(result["passed"])
            self.assertFalse(report.passed)
            self.assertEqual(report.registration_warning_level, "severe")

    def test_validate_step5_detects_outside_mask_diff_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase4_ready_page(project_root)
            result = compose_final(project_root, "0001")
            self.assertTrue(result["passed"])

            page_manifest = load_page_manifest(project_root, "0001")
            final_path = project_root / page_manifest.final_composite_path
            final_image = cv2.imread(str(final_path), cv2.IMREAD_COLOR)
            final_image[:24, :24] = (0, 255, 0)
            cv2.imwrite(str(final_path), final_image)
            cv2.imwrite(str(project_root / page_manifest.result_path), final_image)

            report = validate_step5(project_root, "0001")
            self.assertFalse(report.passed)
            self.assertGreater(report.outside_mask_diff_ratio, 0.005)


if __name__ == "__main__":
    unittest.main()
