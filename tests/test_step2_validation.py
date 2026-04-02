from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import comic_pipeline  # noqa: F401
import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from comic_pipeline.make_balloons_only import make_balloons_only, validate_step2
from comic_pipeline.project import create_project, load_page_manifest, save_page_manifest, scan_pages


class Step2ValidationTests(unittest.TestCase):
    def test_validation_fails_for_wrong_size_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            create_project(project_root)

            image = np.full((60, 90, 3), 130, dtype=np.uint8)
            cv2.imwrite(str(project_root / "0001.png"), image)
            scan_pages(project_root)

            union_mask = np.zeros((60, 90), dtype=np.uint8)
            cv2.rectangle(union_mask, (20, 15), (70, 45), 255, -1)
            mask_path = project_root / "artifacts" / "masks" / "0001_union.png"
            cv2.imwrite(str(mask_path), union_mask)

            page = load_page_manifest(project_root, "0001")
            page.balloon_union_mask_path = str(mask_path.relative_to(project_root))
            page.status = "mask_ready"
            page.balloons_only_rgba_path = "artifacts/layers/0001_balloons_only.png"
            save_page_manifest(project_root, page)

            wrong_size_layer = np.zeros((30, 30, 4), dtype=np.uint8)
            cv2.imwrite(str(project_root / page.balloons_only_rgba_path), wrong_size_layer)

            report = validate_step2(project_root, "0001")
            saved_page = load_page_manifest(project_root, "0001")

            self.assertFalse(report.passed)
            self.assertFalse(report.size_matches)
            self.assertEqual(saved_page.status, "check")

    def test_validation_fails_for_missing_alpha_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            create_project(project_root)

            image = np.full((60, 90, 3), 130, dtype=np.uint8)
            cv2.imwrite(str(project_root / "0001.png"), image)
            scan_pages(project_root)

            union_mask = np.zeros((60, 90), dtype=np.uint8)
            cv2.rectangle(union_mask, (20, 15), (70, 45), 255, -1)
            mask_path = project_root / "artifacts" / "masks" / "0001_union.png"
            cv2.imwrite(str(mask_path), union_mask)

            page = load_page_manifest(project_root, "0001")
            page.balloon_union_mask_path = str(mask_path.relative_to(project_root))
            page.status = "mask_ready"
            save_page_manifest(project_root, page)

            make_balloons_only(project_root, "0001")
            saved_page = load_page_manifest(project_root, "0001")
            layer_path = project_root / saved_page.balloons_only_rgba_path
            broken_layer = np.zeros((60, 90, 3), dtype=np.uint8)
            cv2.imwrite(str(layer_path), broken_layer)

            report = validate_step2(project_root, "0001")
            refreshed_page = load_page_manifest(project_root, "0001")

            self.assertFalse(report.passed)
            self.assertFalse(report.has_alpha_channel)
            self.assertEqual(refreshed_page.status, "check")


if __name__ == "__main__":
    unittest.main()
