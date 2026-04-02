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
from comic_pipeline.step3 import import_external_result, make_handoff, validate_step3
from step3_test_utils import build_phase2_ready_page


class Step3ManualTests(unittest.TestCase):
    def test_make_handoff_writes_package_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase2_ready_page(project_root)

            result = make_handoff(project_root, "0001")
            page_manifest = load_page_manifest(project_root, "0001")
            request_dir = project_root / page_manifest.nano_request_dir

            self.assertTrue(result["passed"])
            self.assertTrue((request_dir / "prompt.md").exists())
            self.assertTrue((request_dir / "request.json").exists())
            self.assertTrue((request_dir / "balloons_only_rgba.png").exists())
            self.assertTrue((request_dir / "original_page.png").exists())

            prompt_text = (request_dir / "prompt.md").read_text(encoding="utf-8")
            self.assertIn("Detect the source language automatically", prompt_text)
            self.assertIn("Translate all speech-bubble text into natural 한국어", prompt_text)
            self.assertIn("Keep everything outside the balloons transparent", prompt_text)

            request_payload = json.loads((request_dir / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(request_payload["expected_output"]["filename"], "0001_raw.png")
            self.assertEqual(request_payload["target_language"], "한국어")
            self.assertEqual(page_manifest.status, "nano_pending")

    def test_import_external_result_normalizes_opaque_output_and_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase2_ready_page(project_root)

            external = np.full((240, 180, 3), 255, dtype=np.uint8)
            cv2.ellipse(external, (90, 95), (32, 20), 0, 0, 360, (170, 170, 170), -1)
            input_path = project_root / "external_result.jpg"
            cv2.imwrite(str(input_path), external)

            result = import_external_result(project_root, "0001", input_path)
            report = validate_step3(project_root, "0001")
            page_manifest = load_page_manifest(project_root, "0001")

            self.assertTrue(result["passed"])
            self.assertTrue(report.passed)
            self.assertTrue(report.opaque_output)
            self.assertFalse(report.has_alpha_channel)
            self.assertEqual(page_manifest.nano_source_kind, "manual_web")
            self.assertEqual(page_manifest.status, "nano_pending")
            self.assertTrue(page_manifest.nano_manual_import_path.endswith(".jpg"))
            self.assertTrue(page_manifest.nano_banana_raw_path.endswith("_raw.png"))
            self.assertTrue((project_root / page_manifest.nano_banana_raw_path).exists())

    def test_import_external_result_rejects_wrong_size_and_marks_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase2_ready_page(project_root)

            wrong_size = np.full((120, 120, 3), 230, dtype=np.uint8)
            input_path = project_root / "external_wrong_size.png"
            cv2.imwrite(str(input_path), wrong_size)

            result = import_external_result(project_root, "0001", input_path)
            page_manifest = load_page_manifest(project_root, "0001")
            report_payload = read_json(project_root / page_manifest.step3_validation_report_path)

            self.assertFalse(result["passed"])
            self.assertEqual(page_manifest.status, "check")
            self.assertEqual(page_manifest.nano_banana_raw_path, "")
            self.assertFalse(report_payload["passed"])
            self.assertIn("size does not match", " ".join(report_payload["notes"]))


if __name__ == "__main__":
    unittest.main()
