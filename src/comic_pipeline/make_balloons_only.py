from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from comic_pipeline.project import (
    load_page_manifest,
    load_project_manifest,
    save_page_manifest,
    save_project_manifest,
    write_json,
)
from comic_pipeline.types import RuntimeDependencyError, Step2ValidationReport


def _require_cv_runtime() -> tuple[object, object]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError(
            "Step 2 runtime requires numpy and opencv-python. "
            "Install the dependencies from pyproject.toml before running make-layer."
        ) from exc
    return cv2, np


def build_soft_alpha_from_union_mask(union_mask: Any) -> tuple[Any, Any]:
    cv2, np = _require_cv_runtime()
    if len(union_mask.shape) != 2:
        raise ValueError("union_mask must be a single-channel image")

    binary_mask = np.where(union_mask > 0, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    support = cv2.dilate(binary_mask, kernel, iterations=1)
    core = cv2.erode(binary_mask, kernel, iterations=1)

    if int(np.count_nonzero(core)) == 0 and int(np.count_nonzero(support)) > 0:
        core = support.copy()

    alpha = cv2.GaussianBlur(support, (5, 5), sigmaX=0.85, sigmaY=0.85)
    alpha[support == 0] = 0
    alpha[core > 0] = 255
    return alpha.astype(np.uint8), support


def compose_balloons_only_rgba(image: Any, alpha: Any) -> Any:
    cv2, np = _require_cv_runtime()
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.shape[:2] != alpha.shape[:2]:
        raise ValueError("image and alpha must have the same height and width")

    rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    active_pixels = alpha > 0
    rgba[:, :, 3] = alpha
    rgba[active_pixels, :3] = image[active_pixels]
    return rgba


def _load_page_inputs(project_root: Path, page_id: str) -> tuple[Any, Any, object]:
    cv2, _ = _require_cv_runtime()
    page_manifest = load_page_manifest(project_root, page_id)
    image_path = project_root / page_manifest.original_path
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read original image: {image_path}")

    if not page_manifest.balloon_union_mask_path:
        raise FileNotFoundError(
            f"Page {page_id} has no saved union mask. Run detect first."
        )

    mask_path = project_root / page_manifest.balloon_union_mask_path
    union_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if union_mask is None:
        raise FileNotFoundError(f"Unable to read union mask: {mask_path}")

    if image.shape[:2] != union_mask.shape[:2]:
        raise ValueError(
            "The original page size does not match the union mask size. "
            f"original={image.shape[:2]}, mask={union_mask.shape[:2]}"
        )

    return image, union_mask, page_manifest


def summarize_step2_validation(
    *,
    page_id: str,
    size_matches: bool,
    has_alpha_channel: bool,
    outside_alpha_sum: int,
    inside_alpha_preservation_ratio: float,
    nonzero_alpha_pixels: int,
    outside_rgb_nonzero_pixels: int,
    min_inside_alpha_preservation_ratio: float = 0.90,
) -> Step2ValidationReport:
    notes: list[str] = []
    if not size_matches:
        notes.append("layer size does not match the original page size")
    if not has_alpha_channel:
        notes.append("layer is missing an alpha channel")
    if outside_alpha_sum != 0:
        notes.append(f"outside-support alpha sum must be 0, got {outside_alpha_sum}")
    if outside_rgb_nonzero_pixels != 0:
        notes.append(
            "outside-support RGB must be fully zeroed, "
            f"got {outside_rgb_nonzero_pixels} leaking pixel(s)"
        )
    if inside_alpha_preservation_ratio < min_inside_alpha_preservation_ratio:
        notes.append(
            "inside-mask alpha preservation ratio is below the minimum threshold: "
            f"{inside_alpha_preservation_ratio:.4f} < {min_inside_alpha_preservation_ratio:.4f}"
        )
    if nonzero_alpha_pixels <= 0:
        notes.append("layer has no nonzero alpha pixels")

    passed = (
        size_matches
        and has_alpha_channel
        and outside_alpha_sum == 0
        and outside_rgb_nonzero_pixels == 0
        and inside_alpha_preservation_ratio >= min_inside_alpha_preservation_ratio
        and nonzero_alpha_pixels > 0
    )
    if passed:
        notes.append("step2 validation passed and the page is ready for Nano Banana")

    return Step2ValidationReport(
        page_id=page_id,
        size_matches=size_matches,
        has_alpha_channel=has_alpha_channel,
        outside_alpha_sum=outside_alpha_sum,
        inside_alpha_preservation_ratio=inside_alpha_preservation_ratio,
        nonzero_alpha_pixels=nonzero_alpha_pixels,
        outside_rgb_nonzero_pixels=outside_rgb_nonzero_pixels,
        passed=passed,
        notes=notes,
    )


def validate_step2(project_root: Path, page_id: str) -> Step2ValidationReport:
    cv2, np = _require_cv_runtime()
    image, union_mask, page_manifest = _load_page_inputs(project_root, page_id)
    if not page_manifest.balloons_only_rgba_path:
        raise FileNotFoundError(
            f"Page {page_id} has no Step 2 output yet. Run make-layer first."
        )

    layer_path = project_root / page_manifest.balloons_only_rgba_path
    layer = cv2.imread(str(layer_path), cv2.IMREAD_UNCHANGED)
    if layer is None:
        raise FileNotFoundError(f"Unable to read Step 2 layer: {layer_path}")

    size_matches = layer.shape[:2] == image.shape[:2]
    has_alpha_channel = len(layer.shape) == 3 and layer.shape[2] == 4
    outside_alpha_sum = 0
    inside_alpha_preservation_ratio = 0.0
    nonzero_alpha_pixels = 0
    outside_rgb_nonzero_pixels = 0

    if size_matches and has_alpha_channel:
        alpha = layer[:, :, 3]
        rgb = layer[:, :, :3]
        _, support = build_soft_alpha_from_union_mask(union_mask)
        outside_support = support == 0
        outside_alpha_sum = int(alpha[outside_support].sum())
        outside_rgb = rgb[outside_support]
        if len(outside_rgb) > 0:
            outside_rgb_nonzero_pixels = int(
                np.count_nonzero(np.any(outside_rgb > 0, axis=1))
            )

        inside_original = union_mask > 0
        inside_original_pixels = int(np.count_nonzero(inside_original))
        if inside_original_pixels > 0:
            inside_alpha_preservation_ratio = float(
                np.count_nonzero(alpha[inside_original] > 0) / inside_original_pixels
            )
        nonzero_alpha_pixels = int(np.count_nonzero(alpha > 0))

    report = summarize_step2_validation(
        page_id=page_id,
        size_matches=size_matches,
        has_alpha_channel=has_alpha_channel,
        outside_alpha_sum=outside_alpha_sum,
        inside_alpha_preservation_ratio=inside_alpha_preservation_ratio,
        nonzero_alpha_pixels=nonzero_alpha_pixels,
        outside_rgb_nonzero_pixels=outside_rgb_nonzero_pixels,
    )

    report_path = project_root / "artifacts" / "debug" / f"{page_id}_step2_validation.json"
    write_json(report_path, report.to_dict())
    page_manifest.step2_validation_report_path = str(report_path.relative_to(project_root))
    page_manifest.status = "nano_pending" if report.passed else "check"
    save_page_manifest(project_root, page_manifest)
    return report


def make_balloons_only(project_root: Path, page_id: str) -> dict[str, Any]:
    cv2, _ = _require_cv_runtime()
    image, union_mask, page_manifest = _load_page_inputs(project_root, page_id)
    alpha, _ = build_soft_alpha_from_union_mask(union_mask)
    rgba_layer = compose_balloons_only_rgba(image, alpha)

    layer_path = project_root / "artifacts" / "layers" / f"{page_id}_balloons_only.png"
    layer_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(layer_path), rgba_layer):
        raise OSError(f"Unable to write Step 2 layer: {layer_path}")

    page_manifest.balloons_only_rgba_path = str(layer_path.relative_to(project_root))
    save_page_manifest(project_root, page_manifest)
    report = validate_step2(project_root, page_id)

    project_manifest = load_project_manifest(project_root)
    project_manifest.last_run_id = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    save_project_manifest(project_root, project_manifest)

    final_manifest = load_page_manifest(project_root, page_id)
    return {
        "page_id": page_id,
        "balloons_only_rgba_path": final_manifest.balloons_only_rgba_path,
        "step2_validation_report_path": final_manifest.step2_validation_report_path,
        "status": final_manifest.status,
        "passed": report.passed,
    }


def format_step2_report(report: Step2ValidationReport) -> str:
    payload = report.to_dict()
    return json.dumps(payload, indent=2, ensure_ascii=False)
