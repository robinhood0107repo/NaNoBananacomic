from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from comic_pipeline.project import (
    load_page_manifest,
    load_project_manifest,
    read_json,
    save_page_manifest,
    save_project_manifest,
    write_json,
)
from comic_pipeline.types import RuntimeDependencyError, Step5ValidationReport

SEAM_BAND_DILATION_PIXELS = 2
DIFF_THRESHOLD = 8
WARNING_DIFF_RATIO = 0.001
FAIL_DIFF_RATIO = 0.005


def _require_step5_runtime() -> tuple[object, object]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError(
            "Step 5 runtime requires numpy and opencv-python. "
            "Install the dependencies from pyproject.toml before running Step 5 commands."
        ) from exc
    return cv2, np


def _project_timestamp(project_root: Path) -> None:
    project_manifest = load_project_manifest(project_root)
    project_manifest.last_run_id = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    save_project_manifest(project_root, project_manifest)


def _as_relative(project_root: Path, path: Path) -> str:
    return str(path.relative_to(project_root))


def _load_step5_context(project_root: Path, page_id: str) -> dict[str, Any]:
    cv2, _ = _require_step5_runtime()
    page_manifest = load_page_manifest(project_root, page_id)
    original_path = project_root / page_manifest.original_path
    overlay_path = project_root / page_manifest.nano_banana_rgba_path if page_manifest.nano_banana_rgba_path else None
    union_mask_path = project_root / page_manifest.balloon_union_mask_path if page_manifest.balloon_union_mask_path else None

    if not original_path.exists():
        raise FileNotFoundError(f"Original page does not exist: {original_path}")
    if overlay_path is None or not overlay_path.exists():
        raise FileNotFoundError(
            f"Page {page_id} has no Step 4 RGBA output yet. Run restore-alpha first."
        )
    if union_mask_path is None or not union_mask_path.exists():
        raise FileNotFoundError(f"Union mask does not exist: {union_mask_path}")

    original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)
    union_mask = cv2.imread(str(union_mask_path), cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Unable to read original page: {original_path}")
    if overlay is None:
        raise FileNotFoundError(f"Unable to read restored overlay: {overlay_path}")
    if union_mask is None:
        raise FileNotFoundError(f"Unable to read union mask: {union_mask_path}")

    return {
        "page_manifest": page_manifest,
        "original": original,
        "overlay": overlay,
        "union_mask": union_mask,
        "original_path": original_path,
        "overlay_path": overlay_path,
    }


def _alpha_composite(original: Any, overlay: Any) -> Any:
    _, np = _require_step5_runtime()
    if overlay.ndim != 3 or overlay.shape[2] != 4:
        raise ValueError("Step 5 overlay must be RGBA")
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    overlay_rgb = overlay[:, :, :3].astype(np.float32)
    base = original.astype(np.float32)
    composed = (overlay_rgb * alpha) + (base * (1.0 - alpha))
    return np.clip(composed, 0, 255).astype(np.uint8)


def _build_diff_preview(original: Any, final: Any, diff_mask: Any, seam_band: Any) -> Any:
    preview = original.copy()
    preview[seam_band > 0] = (0, 180, 255)
    preview[diff_mask > 0] = (20, 20, 240)
    return preview


def summarize_step5_validation(
    *,
    page_id: str,
    final_exists: bool,
    final_size_matches: bool,
    overlay_exists: bool,
    overlay_size_matches: bool,
    registration_report_exists: bool,
    registration_score: float,
    registration_warning_level: str,
    outside_mask_diff_pixels: int,
    outside_mask_region_pixels: int,
    outside_mask_diff_ratio: float,
) -> Step5ValidationReport:
    notes: list[str] = []
    if not final_exists:
        notes.append("final composite output is missing")
    if not final_size_matches:
        notes.append("final composite size does not match the original page size")
    if not overlay_exists:
        notes.append("restored overlay output is missing")
    if not overlay_size_matches:
        notes.append("restored overlay size does not match the original page size")
    if not registration_report_exists:
        notes.append("registration report is missing")
    else:
        notes.append(
            "registration summary: "
            f"level={registration_warning_level}, score={registration_score:.4f}"
        )
    if outside_mask_diff_ratio > FAIL_DIFF_RATIO:
        notes.append(
            "outside-mask diff ratio exceeds the failure threshold: "
            f"{outside_mask_diff_ratio:.6f} > {FAIL_DIFF_RATIO:.6f}"
        )
    elif outside_mask_diff_ratio > WARNING_DIFF_RATIO:
        notes.append(
            "outside-mask diff ratio exceeds the warning threshold: "
            f"{outside_mask_diff_ratio:.6f} > {WARNING_DIFF_RATIO:.6f}"
        )

    passed = (
        final_exists
        and final_size_matches
        and overlay_exists
        and overlay_size_matches
        and registration_report_exists
        and registration_warning_level != "severe"
        and outside_mask_diff_ratio <= FAIL_DIFF_RATIO
    )
    if passed:
        notes.append("step5 validation passed and the final composite is ready for delivery")

    return Step5ValidationReport(
        page_id=page_id,
        final_exists=final_exists,
        final_size_matches=final_size_matches,
        overlay_exists=overlay_exists,
        overlay_size_matches=overlay_size_matches,
        registration_report_exists=registration_report_exists,
        registration_score=registration_score,
        registration_warning_level=registration_warning_level,
        outside_mask_diff_pixels=outside_mask_diff_pixels,
        outside_mask_region_pixels=outside_mask_region_pixels,
        outside_mask_diff_ratio=outside_mask_diff_ratio,
        passed=passed,
        notes=notes,
    )


def validate_step5(project_root: Path, page_id: str) -> Step5ValidationReport:
    cv2, np = _require_step5_runtime()
    context = _load_step5_context(project_root, page_id)
    page_manifest = context["page_manifest"]

    final_path = project_root / page_manifest.final_composite_path if page_manifest.final_composite_path else None
    overlay_path = project_root / page_manifest.nano_banana_rgba_path if page_manifest.nano_banana_rgba_path else None
    registration_path = project_root / page_manifest.registration_report_path if page_manifest.registration_report_path else None

    original = context["original"]
    union_mask = context["union_mask"]

    final_exists = final_path is not None and final_path.exists()
    overlay_exists = overlay_path is not None and overlay_path.exists()
    registration_report_exists = registration_path is not None and registration_path.exists()

    final_size_matches = False
    overlay_size_matches = False
    outside_mask_diff_pixels = 0
    outside_mask_region_pixels = 0
    outside_mask_diff_ratio = 0.0
    registration_score = 0.0
    registration_warning_level = "severe" if not registration_report_exists else "normal"

    if registration_report_exists:
        registration_payload = read_json(registration_path)
        registration_score = float(registration_payload.get("final_score", 0.0))
        registration_warning_level = str(registration_payload.get("warning_level", "severe"))

    final_image = None
    overlay = None
    if final_exists:
        final_image = cv2.imread(str(final_path), cv2.IMREAD_COLOR)
        final_size_matches = (
            final_image is not None
            and final_image.shape[:2] == original.shape[:2]
        )
    if overlay_exists:
        overlay = cv2.imread(str(overlay_path), cv2.IMREAD_UNCHANGED)
        overlay_size_matches = (
            overlay is not None
            and overlay.shape[:2] == original.shape[:2]
        )

    diff_preview_path = project_root / "artifacts" / "previews" / f"{page_id}_diff_preview.png"
    if final_image is not None and final_size_matches:
        kernel_size = (SEAM_BAND_DILATION_PIXELS * 2) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        seam_band = cv2.dilate(np.where(union_mask > 0, 255, 0).astype(np.uint8), kernel, iterations=1)
        outside_mask = seam_band == 0
        absdiff = cv2.absdiff(original, final_image)
        diff_pixels = np.any(absdiff > DIFF_THRESHOLD, axis=2)
        masked_diff = diff_pixels & outside_mask
        outside_mask_diff_pixels = int(np.count_nonzero(masked_diff))
        outside_mask_region_pixels = int(np.count_nonzero(outside_mask))
        if outside_mask_region_pixels > 0:
            outside_mask_diff_ratio = outside_mask_diff_pixels / float(outside_mask_region_pixels)
        diff_preview = _build_diff_preview(original, final_image, masked_diff.astype(np.uint8) * 255, seam_band)
        diff_preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(diff_preview_path), diff_preview)

    report = summarize_step5_validation(
        page_id=page_id,
        final_exists=final_exists,
        final_size_matches=final_size_matches,
        overlay_exists=overlay_exists,
        overlay_size_matches=overlay_size_matches,
        registration_report_exists=registration_report_exists,
        registration_score=registration_score,
        registration_warning_level=registration_warning_level,
        outside_mask_diff_pixels=outside_mask_diff_pixels,
        outside_mask_region_pixels=outside_mask_region_pixels,
        outside_mask_diff_ratio=outside_mask_diff_ratio,
    )

    report_path = project_root / "artifacts" / "debug" / f"{page_id}_step5_validation.json"
    write_json(report_path, report.to_dict())
    page_manifest.step5_validation_report_path = _as_relative(project_root, report_path)
    if diff_preview_path.exists():
        page_manifest.diff_preview_path = _as_relative(project_root, diff_preview_path)
    page_manifest.status = "done" if report.passed else "check"
    save_page_manifest(project_root, page_manifest)
    return report


def compose_final(project_root: Path, page_id: str) -> dict[str, Any]:
    cv2, _ = _require_step5_runtime()
    context = _load_step5_context(project_root, page_id)
    page_manifest = context["page_manifest"]
    final = _alpha_composite(context["original"], context["overlay"])

    composite_path = project_root / "artifacts" / "composite" / f"{page_id}_final.png"
    result_path = project_root / page_manifest.result_path
    composite_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(composite_path), final):
        raise OSError(f"Unable to write final composite: {composite_path}")
    if not cv2.imwrite(str(result_path), final):
        raise OSError(f"Unable to write result image: {result_path}")

    page_manifest.final_composite_path = _as_relative(project_root, composite_path)
    page_manifest.result_path = _as_relative(project_root, result_path)
    save_page_manifest(project_root, page_manifest)

    report = validate_step5(project_root, page_id)
    _project_timestamp(project_root)
    final_manifest = load_page_manifest(project_root, page_id)
    return {
        "page_id": page_id,
        "final_composite_path": final_manifest.final_composite_path,
        "result_path": final_manifest.result_path,
        "diff_preview_path": final_manifest.diff_preview_path,
        "step5_validation_report_path": final_manifest.step5_validation_report_path,
        "status": final_manifest.status,
        "passed": report.passed,
    }


def format_step5_report(report: Step5ValidationReport) -> str:
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
