from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import comic_pipeline  # noqa: F401
import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from comic_pipeline.project import (
    load_page_manifest,
    load_project_manifest,
    read_json,
    save_project_manifest,
)
from comic_pipeline.step3 import clear_session_api_key, run_external_edit, set_session_api_key
from step3_test_utils import build_phase2_ready_page


class _FakeGeminiAdapter:
    def generate(
        self,
        *,
        prompt: str,
        layer_path: Path,
        original_page_path: Path | None,
        output_dir: Path,
        api_key: str,
        model_name: str,
    ) -> Path:
        del original_page_path, api_key, model_name
        if "Keep everything outside the balloons transparent" not in prompt:
            raise AssertionError("Step 3 prompt lost a core constraint")
        if "Use horizontal writing mode for the translated replacement text" not in prompt:
            raise AssertionError("Step 3 prompt lost the horizontal writing constraint")
        image = cv2.imread(str(layer_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise AssertionError("Handoff layer was not created before API execution")
        output_path = output_dir / "fake_response.png"
        cv2.imwrite(str(output_path), image)
        return output_path


class Step3ApiTests(unittest.TestCase):
    def test_run_external_edit_uses_mock_adapter_and_writes_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase2_ready_page(project_root)

            project_manifest = load_project_manifest(project_root)
            project_manifest.nano_integration_mode = "api_auto"
            save_project_manifest(project_root, project_manifest)
            set_session_api_key("gemini", "test-key")

            with patch("comic_pipeline.step3.build_provider_adapter", return_value=_FakeGeminiAdapter()):
                result = run_external_edit(project_root, "0001")

            page_manifest = load_page_manifest(project_root, "0001")
            report_payload = read_json(project_root / page_manifest.step3_validation_report_path)

            self.assertTrue(result["passed"])
            self.assertEqual(page_manifest.nano_source_kind, "api_auto")
            self.assertEqual(page_manifest.status, "nano_pending")
            self.assertTrue((project_root / page_manifest.nano_banana_raw_path).exists())
            self.assertTrue(report_payload["passed"])
            clear_session_api_key("gemini")

    def test_run_external_edit_without_credential_marks_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase2_ready_page(project_root)

            project_manifest = load_project_manifest(project_root)
            project_manifest.nano_integration_mode = "api_auto"
            save_project_manifest(project_root, project_manifest)
            clear_session_api_key("gemini")

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GEMINI_API_KEY", None)
                result = run_external_edit(project_root, "0001")

            page_manifest = load_page_manifest(project_root, "0001")
            report_payload = read_json(project_root / page_manifest.step3_validation_report_path)

            self.assertFalse(result["passed"])
            self.assertEqual(page_manifest.status, "check")
            self.assertEqual(page_manifest.nano_banana_raw_path, "")
            self.assertIn("credential", " ".join(report_payload["notes"]).lower())


if __name__ == "__main__":
    unittest.main()
