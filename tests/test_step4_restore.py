from __future__ import annotations

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

from comic_pipeline.project import load_page_manifest, read_json, save_page_manifest
from comic_pipeline.step3 import import_external_result
from comic_pipeline.step4 import restore_alpha, validate_step4
from step3_test_utils import build_phase2_ready_page


class Step4RestoreTests(unittest.TestCase):
    def test_restore_alpha_from_size_matched_raw_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase2_ready_page(project_root)

            external = np.full((240, 180, 3), 245, dtype=np.uint8)
            cv2.ellipse(external, (90, 95), (32, 20), 0, 0, 360, (180, 180, 180), -1)
            cv2.putText(
                external,
                "KO",
                (72, 101),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )
            input_path = project_root / "external_result.png"
            cv2.imwrite(str(input_path), external)
            imported = import_external_result(project_root, "0001", input_path)
            self.assertTrue(imported["passed"])

            result = restore_alpha(project_root, "0001")
            report = validate_step4(project_root, "0001")
            page_manifest = load_page_manifest(project_root, "0001")
            restored = cv2.imread(
                str(project_root / page_manifest.nano_banana_rgba_path),
                cv2.IMREAD_UNCHANGED,
            )

            self.assertTrue(result["passed"])
            self.assertTrue(report.passed)
            self.assertEqual(page_manifest.status, "restored")
            self.assertEqual(restored.shape, (240, 180, 4))
            self.assertEqual(report.outside_alpha_sum, 0)
            self.assertEqual(report.outside_rgb_nonzero_pixels, 0)
            self.assertGreater(report.nonzero_alpha_pixels, 0)

    def test_restore_alpha_from_manual_import_resizes_and_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase2_ready_page(project_root)

            manual = np.full((300, 220, 3), 238, dtype=np.uint8)
            cv2.rectangle(manual, (20, 30), (200, 270), (210, 210, 210), -1)
            cv2.putText(
                manual,
                "HORIZONTAL",
                (28, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (70, 70, 70),
                2,
                cv2.LINE_AA,
            )
            manual_path = project_root / "imports" / "nano" / "0001_balloons_only.png"
            cv2.imwrite(str(manual_path), manual)

            result = restore_alpha(project_root, "0001")
            report = validate_step4(project_root, "0001")
            page_manifest = load_page_manifest(project_root, "0001")

            self.assertTrue(result["passed"])
            self.assertTrue(report.passed)
            self.assertFalse(report.source_size_matches)
            self.assertTrue(report.alignment_applied)
            self.assertEqual(page_manifest.nano_source_kind, "manual_web")
            self.assertEqual(page_manifest.nano_manual_import_path, "imports/nano/0001_balloons_only.png")
            self.assertTrue(page_manifest.nano_banana_raw_path.endswith("_raw.png"))
            self.assertTrue(page_manifest.nano_banana_rgba_path.endswith("_rgba.png"))

    def test_restore_alpha_without_any_source_marks_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase2_ready_page(project_root)

            result = restore_alpha(project_root, "0001")
            page_manifest = load_page_manifest(project_root, "0001")
            report_payload = read_json(project_root / page_manifest.step4_validation_report_path)

            self.assertFalse(result["passed"])
            self.assertEqual(page_manifest.status, "check")
            self.assertEqual(page_manifest.nano_banana_rgba_path, "")
            self.assertFalse(report_payload["passed"])
            self.assertIn("No Phase 4 input was found", " ".join(report_payload["notes"]))

    def test_restore_alpha_prefers_manual_import_over_existing_raw_for_manual_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase2_ready_page(project_root)

            stale_raw = np.full((240, 180, 3), 30, dtype=np.uint8)
            stale_raw_path = project_root / "artifacts" / "nano" / "0001_raw.png"
            cv2.imwrite(str(stale_raw_path), stale_raw)

            manual = np.full((240, 180, 3), 220, dtype=np.uint8)
            manual_path = project_root / "imports" / "nano" / "0001_submitted.png"
            cv2.imwrite(str(manual_path), manual)

            page_manifest = load_page_manifest(project_root, "0001")
            page_manifest.nano_source_kind = "manual_web"
            page_manifest.nano_manual_import_path = "imports/nano/0001_submitted.png"
            page_manifest.nano_banana_raw_path = "artifacts/nano/0001_raw.png"
            save_page_manifest(project_root, page_manifest)

            result = restore_alpha(project_root, "0001")
            updated_manifest = load_page_manifest(project_root, "0001")
            updated_raw = cv2.imread(
                str(project_root / updated_manifest.nano_banana_raw_path),
                cv2.IMREAD_UNCHANGED,
            )

            self.assertTrue(result["passed"])
            self.assertEqual(updated_manifest.nano_manual_import_path, "imports/nano/0001_submitted.png")
            self.assertEqual(int(updated_raw[0, 0, 0]), 220)

    def test_restore_alpha_cleans_checkerboard_preview_from_opaque_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase2_ready_page(project_root)

            manual = np.full((240, 180, 3), 255, dtype=np.uint8)
            square_a = np.array((188, 188, 188), dtype=np.uint8)
            square_b = np.array((224, 224, 224), dtype=np.uint8)
            for y in range(72, 120, 8):
                for x in range(68, 116, 8):
                    color = square_a if ((x // 8) + (y // 8)) % 2 == 0 else square_b
                    manual[y : y + 8, x : x + 8] = color
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
            manual_path = project_root / "imports" / "nano" / "0001_result.png"
            cv2.imwrite(str(manual_path), manual)

            result = restore_alpha(project_root, "0001")
            report = validate_step4(project_root, "0001")
            page_manifest = load_page_manifest(project_root, "0001")
            cleaned_raw = cv2.imread(
                str(project_root / page_manifest.nano_banana_raw_path),
                cv2.IMREAD_UNCHANGED,
            )

            self.assertTrue(result["passed"])
            self.assertTrue(report.passed)
            self.assertEqual(report.source_mode, "opaque_full_page")
            self.assertTrue(report.checkerboard_cleanup_applied)
            self.assertGreater(report.checkerboard_cleaned_pixels, 0)
            self.assertGreaterEqual(int(cleaned_raw[80, 72, 0]), 245)
            self.assertGreaterEqual(int(cleaned_raw[80, 72, 1]), 245)
            self.assertGreaterEqual(int(cleaned_raw[80, 72, 2]), 245)


if __name__ == "__main__":
    unittest.main()
