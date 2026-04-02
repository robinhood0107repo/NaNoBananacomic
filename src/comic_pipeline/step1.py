from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from comic_pipeline.detectors.contour import ContourBalloonDetector
from comic_pipeline.project import (
    load_page_manifest,
    load_project_manifest,
    save_page_manifest,
    save_project_manifest,
    write_json,
)
from comic_pipeline.types import BalloonPrediction, RuntimeDependencyError, Step1ValidationReport


def _require_cv_runtime() -> tuple[object, object]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError(
            "Step 1 runtime requires numpy and opencv-python. "
            "Install the dependencies from pyproject.toml before running detect."
        ) from exc
    return cv2, np


def _prediction_to_manifest_item(prediction: BalloonPrediction, balloon_id: str) -> dict:
    return {
        "balloon_id": balloon_id,
        "bbox_xyxy": prediction.bbox_xyxy,
        "polygon": prediction.polygon,
        "area": prediction.area,
        "confidence": prediction.confidence,
        "model_name": prediction.model_name,
    }


def build_union_mask(image_shape: tuple[int, int], predictions: Iterable[BalloonPrediction]) -> object:
    cv2, np = _require_cv_runtime()
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for prediction in predictions:
        polygon = np.array(prediction.polygon, dtype=np.int32)
        if polygon.size == 0:
            continue
        cv2.fillPoly(mask, [polygon], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def build_overlay_preview(image: object, union_mask: object) -> object:
    cv2, np = _require_cv_runtime()
    overlay = image.copy()
    color_mask = np.zeros_like(image)
    color_mask[:, :] = (168, 160, 74)
    masked_color = cv2.bitwise_and(color_mask, color_mask, mask=union_mask)
    cv2.addWeighted(masked_color, 0.35, overlay, 1.0, 0.0, overlay)
    return overlay


def summarize_step1_validation(
    *,
    page_id: str,
    width: int,
    height: int,
    predictions: list[BalloonPrediction],
    mask_nonzero_pixels: int,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.45,
) -> Step1ValidationReport:
    notes: list[str] = []
    total_pixels = max(width * height, 1)
    area_ratio = mask_nonzero_pixels / float(total_pixels)
    bbox_out_of_bounds = False
    for prediction in predictions:
        x1, y1, x2, y2 = prediction.bbox_xyxy
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            bbox_out_of_bounds = True
            break

    mask_non_empty = mask_nonzero_pixels > 0 and len(predictions) > 0
    area_ratio_in_expected_range = min_area_ratio <= area_ratio <= max_area_ratio
    if not mask_non_empty:
        notes.append("mask is empty or no balloon predictions were generated")
    if bbox_out_of_bounds:
        notes.append("at least one predicted bbox extends outside the page bounds")
    if not area_ratio_in_expected_range:
        notes.append(
            f"mask area ratio {area_ratio:.4f} is outside the expected range "
            f"[{min_area_ratio:.4f}, {max_area_ratio:.4f}]"
        )

    passed = mask_non_empty and (not bbox_out_of_bounds) and area_ratio_in_expected_range
    if passed:
        notes.append("step1 baseline checks passed")

    return Step1ValidationReport(
        page_id=page_id,
        mask_non_empty=mask_non_empty,
        area_ratio=area_ratio,
        prediction_count=len(predictions),
        bbox_out_of_bounds=bbox_out_of_bounds,
        area_ratio_in_expected_range=area_ratio_in_expected_range,
        passed=passed,
        notes=notes,
    )


def detect_page(project_root: Path, page_id: str) -> dict:
    cv2, np = _require_cv_runtime()
    detector = ContourBalloonDetector()
    page_manifest = load_page_manifest(project_root, page_id)
    image_path = project_root / page_manifest.original_path
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]
    predictions = detector.predict(image)
    union_mask = build_union_mask((height, width), predictions)
    overlay_preview = build_overlay_preview(image, union_mask)

    mask_path = project_root / "artifacts" / "masks" / f"{page_id}_union.png"
    preview_path = project_root / "artifacts" / "previews" / f"{page_id}_overlay.png"
    cv2.imwrite(str(mask_path), union_mask)
    cv2.imwrite(str(preview_path), overlay_preview)

    page_manifest.width = width
    page_manifest.height = height
    page_manifest.balloon_count = len(predictions)
    page_manifest.balloon_union_mask_path = str(mask_path.relative_to(project_root))
    page_manifest.overlay_preview_path = str(preview_path.relative_to(project_root))
    page_manifest.balloons = [
        _prediction_to_manifest_item(prediction, f"{page_id}_b{index:02d}")
        for index, prediction in enumerate(predictions, start=1)
    ]
    page_manifest.status = "mask_ready" if predictions else "check"
    save_page_manifest(project_root, page_manifest)

    project_manifest = load_project_manifest(project_root)
    project_manifest.last_run_id = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    save_project_manifest(project_root, project_manifest)

    return {
        "page_id": page_id,
        "prediction_count": len(predictions),
        "mask_path": str(mask_path),
        "overlay_preview_path": str(preview_path),
        "status": page_manifest.status,
    }


def validate_step1(project_root: Path, page_id: str) -> Step1ValidationReport:
    cv2, _ = _require_cv_runtime()
    page_manifest = load_page_manifest(project_root, page_id)
    if not page_manifest.balloon_union_mask_path:
        raise FileNotFoundError(
            f"Page {page_id} has no saved union mask. Run detect first."
        )

    mask_path = project_root / page_manifest.balloon_union_mask_path
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {mask_path}")

    predictions = [
        BalloonPrediction(
            bbox_xyxy=item["bbox_xyxy"],
            polygon=item["polygon"],
            area=float(item.get("area", 0.0)),
            confidence=float(item.get("confidence", 0.0)),
            model_name=item.get("model_name", "unknown"),
        )
        for item in page_manifest.balloons
    ]
    mask_nonzero_pixels = int((mask > 0).sum())
    report = summarize_step1_validation(
        page_id=page_id,
        width=page_manifest.width,
        height=page_manifest.height,
        predictions=predictions,
        mask_nonzero_pixels=mask_nonzero_pixels,
    )

    report_path = project_root / "artifacts" / "debug" / f"{page_id}_step1_validation.json"
    write_json(report_path, report.to_dict())
    page_manifest.validation_report_path = str(report_path.relative_to(project_root))
    page_manifest.status = page_manifest.status if report.passed else "check"
    save_page_manifest(project_root, page_manifest)
    return report


def format_report(report: Step1ValidationReport) -> str:
    payload = report.to_dict()
    return json.dumps(payload, indent=2, ensure_ascii=False)

