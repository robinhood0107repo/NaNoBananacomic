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

from comic_pipeline.registration import (
    align_source_to_page,
    compute_scale_translation_matrix,
    estimate_support_from_image,
)
from step3_test_utils import build_phase2_ready_page


class RegistrationTests(unittest.TestCase):
    def test_estimate_support_from_rgb_import_detects_balloon_region(self) -> None:
        checker = np.zeros((320, 220, 3), dtype=np.uint8)
        tile = 16
        for y in range(0, checker.shape[0], tile):
            for x in range(0, checker.shape[1], tile):
                value = 220 if ((x // tile) + (y // tile)) % 2 == 0 else 245
                checker[y : y + tile, x : x + tile] = (value, value, value)
        cv2.ellipse(checker, (110, 120), (42, 28), 0, 0, 360, (250, 250, 250), -1)
        cv2.ellipse(checker, (110, 120), (42, 28), 0, 0, 360, (20, 20, 20), 2)

        support, source = estimate_support_from_image(checker)

        self.assertGreater(int(np.count_nonzero(support)), 0)
        self.assertIn(source, {"contour_baseline_v1", "bright_threshold_fallback"})

    def test_compute_scale_translation_matrix_aligns_bbox_and_centroid(self) -> None:
        moving = np.zeros((300, 220), dtype=np.uint8)
        reference = np.zeros((240, 180), dtype=np.uint8)
        cv2.rectangle(moving, (40, 60), (160, 220), 255, -1)
        cv2.rectangle(reference, (30, 40), (120, 170), 255, -1)

        matrix = np.array(compute_scale_translation_matrix(moving, reference), dtype=np.float32)
        warped = cv2.warpAffine(moving, matrix, (180, 240), flags=cv2.INTER_NEAREST)

        intersection = np.count_nonzero((warped > 0) & (reference > 0))
        union = np.count_nonzero((warped > 0) | (reference > 0))
        score = intersection / float(union)
        self.assertGreater(score, 0.9)

    def test_align_source_to_page_prefers_better_than_resize_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            build_phase2_ready_page(project_root)

            source = np.full((300, 220, 3), 235, dtype=np.uint8)
            cv2.ellipse(source, (110, 135), (50, 32), 0, 0, 360, (250, 250, 250), -1)
            cv2.fillConvexPoly(
                source,
                np.array([[132, 152], [178, 195], [118, 186]], dtype=np.int32),
                (250, 250, 250),
            )
            source_path = project_root / "manual.png"
            cv2.imwrite(str(source_path), source)

            report = align_source_to_page(
                project_root=project_root,
                page_id="0001",
                source_path=source_path,
                source_kind="manual_web",
                expected_size=(180, 240),
                raw_output_path=project_root / "artifacts" / "nano" / "0001_raw.png",
                report_output_path=project_root / "artifacts" / "debug" / "0001_registration.json",
                preview_output_path=project_root / "artifacts" / "previews" / "0001_registration_overlay.png",
            )

            self.assertTrue((project_root / "artifacts" / "previews" / "0001_registration_overlay.png").exists())
            self.assertGreaterEqual(report.final_score, report.initial_score)


if __name__ == "__main__":
    unittest.main()
