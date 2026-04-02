from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comic_pipeline.detectors.contour import ContourBalloonDetector
from comic_pipeline.project import load_page_manifest, save_page_manifest, write_json
from comic_pipeline.types import RegistrationReport, RuntimeDependencyError


def _require_registration_runtime() -> tuple[object, object]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError(
            "Registration runtime requires numpy and opencv-python. "
            "Install the dependencies from pyproject.toml before running Step 4/5."
        ) from exc
    return cv2, np


def _as_relative(project_root: Path, path: Path) -> str:
    return str(path.relative_to(project_root))


def _load_reference_support(project_root: Path, page_id: str, expected_size: tuple[int, int]) -> tuple[Any, str]:
    cv2, np = _require_registration_runtime()
    page_manifest = load_page_manifest(project_root, page_id)

    if page_manifest.balloons_only_rgba_path:
        layer_path = project_root / page_manifest.balloons_only_rgba_path
        layer = cv2.imread(str(layer_path), cv2.IMREAD_UNCHANGED)
        if layer is not None and layer.ndim == 3 and layer.shape[2] == 4:
            support = np.where(layer[:, :, 3] > 0, 255, 0).astype(np.uint8)
            if support.shape[::-1] != expected_size:
                support = cv2.resize(support, expected_size, interpolation=cv2.INTER_NEAREST)
            if int(np.count_nonzero(support)) > 0:
                return support, "balloons_only_rgba_alpha"

    if page_manifest.balloon_union_mask_path:
        union_mask_path = project_root / page_manifest.balloon_union_mask_path
        union_mask = cv2.imread(str(union_mask_path), cv2.IMREAD_GRAYSCALE)
        if union_mask is not None:
            support = np.where(union_mask > 0, 255, 0).astype(np.uint8)
            if support.shape[::-1] != expected_size:
                support = cv2.resize(support, expected_size, interpolation=cv2.INTER_NEAREST)
            return support, "balloon_union_mask"

    raise FileNotFoundError(f"Unable to load a reference support mask for page {page_id}")


def _rasterize_predictions(image_shape: tuple[int, int], predictions: list[Any]) -> Any:
    cv2, np = _require_registration_runtime()
    height, width = image_shape
    support = np.zeros((height, width), dtype=np.uint8)
    for prediction in predictions:
        polygon = np.array(prediction.polygon, dtype=np.int32)
        if polygon.size == 0:
            continue
        cv2.fillPoly(support, [polygon], 255)
    return support


def _fallback_bright_support(image: Any) -> Any:
    cv2, np = _require_registration_runtime()
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 185, 255, cv2.THRESH_BINARY)
    flood = binary.copy()
    height, width = flood.shape

    def flood_from_border(seed_x: int, seed_y: int) -> None:
        if flood[seed_y, seed_x] != 255:
            return
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_mask, (seed_x, seed_y), 0)

    for x in range(width):
        flood_from_border(x, 0)
        flood_from_border(x, height - 1)
    for y in range(height):
        flood_from_border(0, y)
        flood_from_border(width - 1, y)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(flood, cv2.MORPH_CLOSE, kernel, iterations=2)


def estimate_support_from_image(image: Any) -> tuple[Any, str]:
    cv2, np = _require_registration_runtime()
    if image.ndim == 3 and image.shape[2] == 4:
        alpha_support = np.where(image[:, :, 3] > 0, 255, 0).astype(np.uint8)
        if int(np.count_nonzero(alpha_support)) > 0:
            return alpha_support, "alpha_channel"

    detector = ContourBalloonDetector()
    bgr = image
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    predictions = detector.predict(bgr)
    if predictions:
        support = _rasterize_predictions(bgr.shape[:2], predictions)
        if int(np.count_nonzero(support)) > 0:
            return support, detector.name

    fallback = _fallback_bright_support(bgr)
    return np.where(fallback > 0, 255, 0).astype(np.uint8), "bright_threshold_fallback"


def _mask_bbox(mask: Any) -> tuple[int, int, int, int]:
    cv2, _ = _require_registration_runtime()
    x, y, width, height = cv2.boundingRect(mask)
    return int(x), int(y), int(width), int(height)


def _mask_centroid(mask: Any) -> tuple[float, float]:
    _, np = _require_registration_runtime()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def compute_scale_translation_matrix(moving_support: Any, reference_support: Any) -> list[list[float]]:
    _, np = _require_registration_runtime()
    if int(np.count_nonzero(moving_support)) == 0 or int(np.count_nonzero(reference_support)) == 0:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    mx, my, mw, mh = _mask_bbox(moving_support)
    rx, ry, rw, rh = _mask_bbox(reference_support)
    m_cx, m_cy = _mask_centroid(moving_support)
    r_cx, r_cy = _mask_centroid(reference_support)

    scale_x = rw / float(max(mw, 1))
    scale_y = rh / float(max(mh, 1))
    translate_x = r_cx - (scale_x * m_cx)
    translate_y = r_cy - (scale_y * m_cy)
    return [
        [float(scale_x), 0.0, float(translate_x)],
        [0.0, float(scale_y), float(translate_y)],
    ]


def _resize_to_canvas_matrix(source_size: tuple[int, int], expected_size: tuple[int, int]) -> list[list[float]]:
    source_width, source_height = source_size
    target_width, target_height = expected_size
    return [
        [target_width / float(max(source_width, 1)), 0.0, 0.0],
        [0.0, target_height / float(max(source_height, 1)), 0.0],
    ]


def _to_affine_array(matrix: list[list[float]]) -> Any:
    _, np = _require_registration_runtime()
    return np.array(matrix, dtype=np.float32)


def _compose_affine(forward_delta: Any, base_forward: Any) -> Any:
    _, np = _require_registration_runtime()
    delta3 = np.vstack([forward_delta, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    base3 = np.vstack([base_forward, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    return (delta3 @ base3)[:2, :].astype(np.float32)


def _warp_affine(image: Any, matrix: Any, expected_size: tuple[int, int], *, is_mask: bool) -> Any:
    cv2, _ = _require_registration_runtime()
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    border_value: Any = 0
    if image.ndim == 3:
        border_value = (0,) * image.shape[2]
    return cv2.warpAffine(
        image,
        matrix,
        expected_size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _mask_iou(reference_support: Any, moving_support: Any) -> float:
    _, np = _require_registration_runtime()
    reference_binary = reference_support > 0
    moving_binary = moving_support > 0
    union = int(np.count_nonzero(reference_binary | moving_binary))
    if union == 0:
        return 0.0
    intersection = int(np.count_nonzero(reference_binary & moving_binary))
    return float(intersection / union)


def _try_ecc_refinement(reference_support: Any, prealigned_support: Any) -> tuple[float | None, Any | None]:
    cv2, np = _require_registration_runtime()
    if int(np.count_nonzero(prealigned_support)) == 0:
        return (None, None)
    template = (reference_support > 0).astype(np.float32)
    moving = (prealigned_support > 0).astype(np.float32)
    warp_inverse = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        50,
        1e-5,
    )
    try:
        score, warp_inverse = cv2.findTransformECC(
            template,
            moving,
            warp_inverse,
            cv2.MOTION_AFFINE,
            criteria,
        )
        forward_delta = cv2.invertAffineTransform(warp_inverse)
    except cv2.error:
        return (None, None)
    return (float(score), forward_delta.astype(np.float32))


@dataclass(slots=True)
class _RegistrationCandidate:
    name: str
    matrix: Any
    score: float


def _render_registration_preview(reference_support: Any, aligned_support: Any, output_path: Path) -> None:
    cv2, np = _require_registration_runtime()
    preview = np.zeros((reference_support.shape[0], reference_support.shape[1], 3), dtype=np.uint8)
    reference_binary = reference_support > 0
    aligned_binary = aligned_support > 0
    preview[reference_binary] = (40, 170, 80)
    preview[aligned_binary] = (50, 80, 210)
    preview[reference_binary & aligned_binary] = (70, 210, 220)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)


def align_source_to_page(
    *,
    project_root: Path,
    page_id: str,
    source_path: Path,
    source_kind: str | None,
    expected_size: tuple[int, int],
    raw_output_path: Path,
    report_output_path: Path,
    preview_output_path: Path,
) -> RegistrationReport:
    cv2, np = _require_registration_runtime()
    source_image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if source_image is None:
        raise FileNotFoundError(f"Unable to read registration source image: {source_path}")

    source_height, source_width = source_image.shape[:2]
    source_size_matches = (source_width, source_height) == expected_size
    reference_support, reference_source = _load_reference_support(project_root, page_id, expected_size)
    moving_support, moving_support_source = estimate_support_from_image(source_image)

    candidates: list[_RegistrationCandidate] = []

    resize_matrix = _to_affine_array(_resize_to_canvas_matrix((source_width, source_height), expected_size))
    resized_support = _warp_affine(moving_support, resize_matrix, expected_size, is_mask=True)
    candidates.append(
        _RegistrationCandidate(
            name="resize_to_canvas",
            matrix=resize_matrix,
            score=_mask_iou(reference_support, resized_support),
        )
    )

    if source_size_matches:
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        candidates.append(
            _RegistrationCandidate(
                name="identity",
                matrix=identity,
                score=_mask_iou(reference_support, _warp_affine(moving_support, identity, expected_size, is_mask=True)),
            )
        )

    initial_matrix = _to_affine_array(compute_scale_translation_matrix(moving_support, reference_support))
    initial_aligned = _warp_affine(moving_support, initial_matrix, expected_size, is_mask=True)
    initial_score = _mask_iou(reference_support, initial_aligned)
    candidates.append(
        _RegistrationCandidate(
            name="support_scale_translation",
            matrix=initial_matrix,
            score=initial_score,
        )
    )

    best_candidate = max(candidates, key=lambda item: item.score)
    used_canvas_match_fallback = False
    if source_size_matches and best_candidate.score < 0.05:
        best_candidate = _RegistrationCandidate(
            name="canvas_match_identity_fallback",
            matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            score=1.0,
        )
        used_canvas_match_fallback = True
    refined_score = best_candidate.score
    if best_candidate.name != "identity" and best_candidate.score < 0.75:
        prealigned_support = _warp_affine(moving_support, best_candidate.matrix, expected_size, is_mask=True)
        ecc_score, forward_delta = _try_ecc_refinement(reference_support, prealigned_support)
        if ecc_score is not None and forward_delta is not None:
            refined_matrix = _compose_affine(forward_delta, best_candidate.matrix)
            refined_support = _warp_affine(moving_support, refined_matrix, expected_size, is_mask=True)
            refined_score = _mask_iou(reference_support, refined_support)
            if refined_score > best_candidate.score:
                best_candidate = _RegistrationCandidate(
                    name="ecc_affine_refine",
                    matrix=refined_matrix,
                    score=refined_score,
                )

    aligned_image = _warp_affine(source_image, best_candidate.matrix, expected_size, is_mask=False)
    aligned_support = _warp_affine(moving_support, best_candidate.matrix, expected_size, is_mask=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(raw_output_path), aligned_image):
        raise OSError(f"Unable to write aligned raw image: {raw_output_path}")

    _render_registration_preview(reference_support, aligned_support, preview_output_path)

    if best_candidate.score >= 0.75:
        warning_level = "normal"
    elif best_candidate.score >= 0.55:
        warning_level = "warning"
    else:
        warning_level = "severe"

    notes: list[str] = []
    if not source_size_matches:
        notes.append("source image size did not match the original page size and was aligned to the page canvas")
    if moving_support_source != "alpha_channel":
        notes.append(f"moving support was estimated from the import using {moving_support_source}")
    if used_canvas_match_fallback:
        notes.append("source canvas already matched the page size, so identity alignment fallback was used")
    if warning_level != "normal":
        notes.append(
            "registration score is below the nominal threshold and the final composite should be reviewed"
        )

    report = RegistrationReport(
        page_id=page_id,
        source_kind=source_kind,
        reference_source=reference_source,
        moving_support_source=moving_support_source,
        source_size_matches=source_size_matches,
        alignment_applied=best_candidate.name not in {"identity", "canvas_match_identity_fallback"} or (not source_size_matches),
        method_used=best_candidate.name,
        initial_score=initial_score,
        refined_score=refined_score,
        final_score=best_candidate.score,
        warning_level=warning_level,
        source_width=source_width,
        source_height=source_height,
        output_width=expected_size[0],
        output_height=expected_size[1],
        transform_matrix=[[float(value) for value in row] for row in best_candidate.matrix.tolist()],
        notes=notes,
    )
    write_json(report_output_path, report.to_dict())

    page_manifest = load_page_manifest(project_root, page_id)
    page_manifest.registration_report_path = _as_relative(project_root, report_output_path)
    save_page_manifest(project_root, page_manifest)
    return report
