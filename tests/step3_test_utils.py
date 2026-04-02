from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import comic_pipeline  # noqa: F401
import cv2  # type: ignore[import-not-found]
import numpy as np  # type: ignore[import-not-found]

from comic_pipeline.make_balloons_only import make_balloons_only
from comic_pipeline.project import create_project, load_page_manifest, save_page_manifest, scan_pages


def build_phase2_ready_page(project_root: Path, page_id: str = "0001") -> None:
    create_project(project_root)
    image = np.full((240, 180, 3), 90, dtype=np.uint8)
    cv2.rectangle(image, (12, 12), (168, 228), (25, 25, 25), 4)
    cv2.ellipse(image, (90, 95), (40, 28), 0, 0, 360, (248, 248, 248), -1)
    cv2.ellipse(image, (90, 95), (40, 28), 0, 0, 360, (15, 15, 15), 3)
    cv2.fillConvexPoly(
        image,
        np.array([[112, 114], [142, 150], [95, 142]], dtype=np.int32),
        (248, 248, 248),
    )
    cv2.polylines(
        image,
        [np.array([[112, 114], [142, 150], [95, 142]], dtype=np.int32)],
        True,
        (15, 15, 15),
        3,
    )
    cv2.imwrite(str(project_root / f"{page_id}.png"), image)

    scan_pages(project_root)

    union_mask = np.zeros((240, 180), dtype=np.uint8)
    cv2.ellipse(union_mask, (90, 95), (40, 28), 0, 0, 360, 255, -1)
    cv2.fillConvexPoly(
        union_mask,
        np.array([[112, 114], [142, 150], [95, 142]], dtype=np.int32),
        255,
    )
    mask_path = project_root / "artifacts" / "masks" / f"{page_id}_union.png"
    cv2.imwrite(str(mask_path), union_mask)

    page_manifest = load_page_manifest(project_root, page_id)
    page_manifest.status = "mask_ready"
    page_manifest.balloon_count = 1
    page_manifest.balloon_union_mask_path = str(mask_path.relative_to(project_root))
    save_page_manifest(project_root, page_manifest)

    result = make_balloons_only(project_root, page_id)
    if not result["passed"]:
        raise AssertionError("Step 2 setup failed in test helper")
