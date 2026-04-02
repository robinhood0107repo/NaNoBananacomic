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

from comic_pipeline.make_balloons_only import (
    build_soft_alpha_from_union_mask,
    compose_balloons_only_rgba,
    make_balloons_only,
)
from comic_pipeline.project import (
    create_project,
    load_page_manifest,
    save_page_manifest,
    scan_pages,
)


class Step2LayerTests(unittest.TestCase):
    def test_compose_rgba_preserves_size_and_zeroes_outside(self) -> None:
        image = np.full((32, 40, 3), (20, 90, 180), dtype=np.uint8)
        union_mask = np.zeros((32, 40), dtype=np.uint8)
        cv2.circle(union_mask, (20, 16), 6, 255, -1)

        alpha, support = build_soft_alpha_from_union_mask(union_mask)
        rgba = compose_balloons_only_rgba(image, alpha)

        self.assertEqual(rgba.shape, (32, 40, 4))
        outside_support = support == 0
        self.assertEqual(int(rgba[:, :, 3][outside_support].sum()), 0)
        outside_rgb = rgba[:, :, :3][outside_support]
        self.assertEqual(
            int(np.count_nonzero(np.any(outside_rgb > 0, axis=1))),
            0,
        )

    def test_tiny_balloon_keeps_nonzero_alpha(self) -> None:
        union_mask = np.zeros((7, 7), dtype=np.uint8)
        union_mask[3, 3] = 255

        alpha, _ = build_soft_alpha_from_union_mask(union_mask)

        self.assertGreater(int(np.count_nonzero(alpha)), 0)
        self.assertGreater(int(alpha[3, 3]), 0)

    def test_make_layer_updates_manifest_and_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            create_project(project_root)
            image = np.full((80, 120, 3), 70, dtype=np.uint8)
            cv2.ellipse(image, (60, 40), (25, 18), 0, 0, 360, (240, 240, 240), -1)
            image_path = project_root / "0001.png"
            cv2.imwrite(str(image_path), image)
            scan_pages(project_root)

            union_mask = np.zeros((80, 120), dtype=np.uint8)
            cv2.ellipse(union_mask, (60, 40), (25, 18), 0, 0, 360, 255, -1)
            mask_path = project_root / "artifacts" / "masks" / "0001_union.png"
            cv2.imwrite(str(mask_path), union_mask)

            page = load_page_manifest(project_root, "0001")
            page.status = "mask_ready"
            page.balloon_union_mask_path = str(mask_path.relative_to(project_root))
            save_page_manifest(project_root, page)

            result = make_balloons_only(project_root, "0001")
            saved_page = load_page_manifest(project_root, "0001")

            self.assertTrue(result["passed"])
            self.assertEqual(saved_page.status, "nano_pending")
            self.assertTrue(saved_page.balloons_only_rgba_path.endswith(".png"))
            self.assertTrue((project_root / saved_page.balloons_only_rgba_path).exists())
            self.assertTrue((project_root / saved_page.step2_validation_report_path).exists())

    def test_make_layer_fails_when_mask_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            create_project(project_root)
            image = np.full((40, 40, 3), 100, dtype=np.uint8)
            image_path = project_root / "0001.png"
            cv2.imwrite(str(image_path), image)
            scan_pages(project_root)

            with self.assertRaises(FileNotFoundError):
                make_balloons_only(project_root, "0001")


if __name__ == "__main__":
    unittest.main()
