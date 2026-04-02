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

from comic_pipeline.project import create_project, load_page_manifest, scan_pages
from comic_pipeline.step1 import detect_page, validate_step1


class Step1DetectSmokeTests(unittest.TestCase):
    def test_detect_and_validate_on_synthetic_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            create_project(project_root)

            image = np.full((1400, 1000, 3), 120, dtype=np.uint8)
            cv2.rectangle(image, (40, 40), (960, 1360), (40, 40, 40), 6)
            cv2.rectangle(image, (60, 440), (940, 720), (70, 70, 70), 3)
            cv2.ellipse(image, (300, 250), (170, 110), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(image, (300, 250), (170, 110), 0, 0, 360, (10, 10, 10), 4)
            cv2.fillConvexPoly(
                image,
                np.array([[430, 310], [500, 380], [380, 360]], dtype=np.int32),
                (255, 255, 255),
            )
            cv2.polylines(
                image,
                [np.array([[430, 310], [500, 380], [380, 360]], dtype=np.int32)],
                True,
                (10, 10, 10),
                4,
            )
            cv2.ellipse(image, (720, 960), (150, 100), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(image, (720, 960), (150, 100), 0, 0, 360, (10, 10, 10), 4)

            image_path = project_root / "0001.png"
            cv2.imwrite(str(image_path), image)

            scan_pages(project_root)
            detect_result = detect_page(project_root, "0001", detector_name="contour_baseline_v1")
            report = validate_step1(project_root, "0001")
            page_manifest = load_page_manifest(project_root, "0001")

            self.assertGreaterEqual(detect_result["prediction_count"], 1)
            self.assertFalse(report.passed)
            self.assertEqual(page_manifest.status, "check")
            self.assertTrue((project_root / page_manifest.balloon_union_mask_path).exists())
            self.assertTrue((project_root / page_manifest.overlay_preview_path).exists())


if __name__ == "__main__":
    unittest.main()
