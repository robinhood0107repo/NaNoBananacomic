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
from comic_pipeline.step1 import validate_step1
from comic_pipeline.step3 import import_external_result
from comic_pipeline.step4 import restore_alpha


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
    page_manifest.step1_detector_name = "manga109_seg_v1"
    page_manifest.balloon_union_mask_path = str(mask_path.relative_to(project_root))
    page_manifest.balloons = [
        {
            "balloon_id": f"{page_id}_b01",
            "bbox_xyxy": [50, 67, 143, 151],
            "polygon": [[50, 95], [55, 78], [71, 67], [109, 67], [130, 81], [143, 114], [95, 142], [90, 123]],
            "area": 4521.0,
            "confidence": 0.98,
            "model_name": "manga109_seg_v1",
        }
    ]
    save_page_manifest(project_root, page_manifest)
    validate_step1(project_root, page_id)

    result = make_balloons_only(project_root, page_id)
    if not result["passed"]:
        raise AssertionError("Step 2 setup failed in test helper")


def build_phase4_ready_page(
    project_root: Path,
    *,
    page_id: str = "0001",
    mismatch_manual_import: bool = False,
) -> None:
    build_phase2_ready_page(project_root, page_id=page_id)

    if mismatch_manual_import:
        external = np.full((300, 220, 3), 238, dtype=np.uint8)
        cv2.rectangle(external, (20, 30), (200, 270), (210, 210, 210), -1)
        cv2.putText(
            external,
            "HORIZONTAL",
            (28, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (70, 70, 70),
            2,
            cv2.LINE_AA,
        )
    else:
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

    input_path = project_root / f"{page_id}_external.png"
    cv2.imwrite(str(input_path), external)
    imported = import_external_result(project_root, page_id, input_path)
    if not imported["passed"]:
        raise AssertionError("Step 3 setup failed in test helper")

    restored = restore_alpha(project_root, page_id)
    if not restored["passed"]:
        raise AssertionError("Step 4 setup failed in test helper")
