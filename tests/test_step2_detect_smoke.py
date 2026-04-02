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
from comic_pipeline.project import create_project, load_page_manifest, scan_pages
from comic_pipeline.step1 import detect_page


class Step2DetectSmokeTests(unittest.TestCase):
    def test_detect_then_make_layer_then_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            create_project(project_root)

            image = np.full((1200, 900, 3), 120, dtype=np.uint8)
            cv2.rectangle(image, (40, 40), (860, 1160), (40, 40, 40), 6)
            cv2.ellipse(image, (280, 240), (160, 100), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(image, (280, 240), (160, 100), 0, 0, 360, (10, 10, 10), 4)
            cv2.fillConvexPoly(
                image,
                np.array([[410, 300], [470, 360], [360, 350]], dtype=np.int32),
                (255, 255, 255),
            )
            cv2.polylines(
                image,
                [np.array([[410, 300], [470, 360], [360, 350]], dtype=np.int32)],
                True,
                (10, 10, 10),
                4,
            )
            cv2.imwrite(str(project_root / "0001.png"), image)

            scan_pages(project_root)
            detect_page(project_root, "0001", detector_name="contour_baseline_v1")
            layer_result = make_balloons_only(project_root, "0001")
            report = validate_step2(project_root, "0001")
            page_manifest = load_page_manifest(project_root, "0001")

            self.assertTrue(layer_result["passed"])
            self.assertTrue(report.passed)
            self.assertEqual(page_manifest.status, "nano_pending")
            self.assertTrue((project_root / page_manifest.balloons_only_rgba_path).exists())


if __name__ == "__main__":
    unittest.main()
