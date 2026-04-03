from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comic_pipeline.detectors.registry import build_detector
from comic_pipeline.detectors.contour import ContourBalloonDetector
from comic_pipeline.project import load_page_manifest, save_page_manifest, write_json
from comic_pipeline.types import BalloonPrediction, RegistrationReport, RuntimeDependencyError


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


@dataclass(slots=True)
class _BalloonRecord:
    balloon_id: str
    bbox_xyxy: tuple[int, int, int, int]
    polygon: list[list[int]]
    mask: Any
    centroid: tuple[float, float]
    area: float


@dataclass(slots=True)
class _BalloonMatch:
    reference: _BalloonRecord
    imported: _BalloonRecord
    polygon_iou: float
    bbox_iou: float
    centroid_proximity: float
    score: float


def _normalize_bbox_xyxy(bbox_xyxy: list[int] | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
    return (x1, y1, x2, y2)


def _bbox_width_height(bbox_xyxy: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return (max(x2 - x1, 1), max(y2 - y1, 1))


def _bbox_iou_xyxy(reference_bbox: tuple[int, int, int, int], moving_bbox: tuple[int, int, int, int]) -> float:
    _, np = _require_registration_runtime()
    rx1, ry1, rx2, ry2 = reference_bbox
    mx1, my1, mx2, my2 = moving_bbox
    ix1 = max(rx1, mx1)
    iy1 = max(ry1, my1)
    ix2 = min(rx2, mx2)
    iy2 = min(ry2, my2)
    iw = max(ix2 - ix1, 0)
    ih = max(iy2 - iy1, 0)
    intersection = iw * ih
    reference_area = max((rx2 - rx1) * (ry2 - ry1), 1)
    moving_area = max((mx2 - mx1) * (my2 - my1), 1)
    union = reference_area + moving_area - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def _centroid_distance(reference: tuple[float, float], moving: tuple[float, float]) -> float:
    _, np = _require_registration_runtime()
    return float(np.hypot(reference[0] - moving[0], reference[1] - moving[1]))


def _centroid_proximity_score(
    reference: tuple[float, float],
    moving: tuple[float, float],
    reference_bbox: tuple[int, int, int, int],
) -> float:
    _, np = _require_registration_runtime()
    width, height = _bbox_width_height(reference_bbox)
    diagonal = float(np.hypot(width, height))
    if diagonal <= 0.0:
        return 0.0
    distance = _centroid_distance(reference, moving)
    normalized = max(0.0, 1.0 - (distance / diagonal))
    return float(min(normalized, 1.0))


def _polygon_to_mask(image_shape: tuple[int, int], polygon: list[list[int]]) -> Any:
    cv2, np = _require_registration_runtime()
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygon:
        return mask
    polygon_array = np.array(polygon, dtype=np.int32)
    if polygon_array.size == 0:
        return mask
    cv2.fillPoly(mask, [polygon_array], 255)
    return mask


def _centroid_from_mask(mask: Any) -> tuple[float, float]:
    return _mask_centroid(mask)


def _build_reference_balloon_records(page_id: str, balloons: list[dict[str, Any]], image_shape: tuple[int, int]) -> list[_BalloonRecord]:
    records: list[_BalloonRecord] = []
    for index, balloon in enumerate(balloons, start=1):
        balloon_id = str(balloon.get("balloon_id") or f"{page_id}_b{index:02d}")
        polygon = [[int(point[0]), int(point[1])] for point in balloon.get("polygon", [])]
        bbox_xyxy = _normalize_bbox_xyxy(balloon.get("bbox_xyxy", [0, 0, 0, 0]))
        mask = _polygon_to_mask(image_shape, polygon)
        area = float(balloon.get("area", 0.0))
        if area <= 0.0:
            area = float((mask > 0).sum())
        records.append(
            _BalloonRecord(
                balloon_id=balloon_id,
                bbox_xyxy=bbox_xyxy,
                polygon=polygon,
                mask=mask,
                centroid=_centroid_from_mask(mask),
                area=area,
            )
        )
    return records


def _build_import_balloon_records(predictions: list[BalloonPrediction], image_shape: tuple[int, int]) -> list[_BalloonRecord]:
    records: list[_BalloonRecord] = []
    for index, prediction in enumerate(predictions, start=1):
        mask = _polygon_to_mask(image_shape, prediction.polygon)
        records.append(
            _BalloonRecord(
                balloon_id=f"import_b{index:02d}",
                bbox_xyxy=_normalize_bbox_xyxy(prediction.bbox_xyxy),
                polygon=[[int(point[0]), int(point[1])] for point in prediction.polygon],
                mask=mask,
                centroid=_centroid_from_mask(mask),
                area=float(prediction.area),
            )
        )
    return records


def _prepare_detector_input(image: Any) -> Any:
    cv2, np = _require_registration_runtime()
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        base = np.full((image.shape[0], image.shape[1], 3), 255.0, dtype=np.float32)
        foreground = image[:, :, :3].astype(np.float32)
        composed = (foreground * alpha) + (base * (1.0 - alpha))
        return np.clip(composed, 0, 255).astype(np.uint8)
    return image[:, :, :3]


def _run_import_detector(image: Any, detector_name: str) -> tuple[list[BalloonPrediction], str, str | None]:
    detector_input = _prepare_detector_input(image)
    try:
        detector = build_detector(detector_name)
        return detector.predict(detector_input), detector.name, None
    except Exception as exc:  # pragma: no cover - exercised through integration fallback
        return [], detector_name, f"detector-based import refinement failed: {exc}"


def _match_detector_balloons(
    reference_records: list[_BalloonRecord],
    import_records: list[_BalloonRecord],
) -> tuple[list[_BalloonMatch], list[str], list[str]]:
    candidates: list[_BalloonMatch] = []
    for reference in reference_records:
        ref_width, ref_height = _bbox_width_height(reference.bbox_xyxy)
        ref_diagonal = max((ref_width**2 + ref_height**2) ** 0.5, 1.0)
        for imported in import_records:
            area_ratio = imported.area / float(max(reference.area, 1.0))
            if area_ratio < 0.25 or area_ratio > 4.0:
                continue
            if _centroid_distance(reference.centroid, imported.centroid) > ref_diagonal:
                continue
            polygon_iou = _mask_iou(reference.mask, imported.mask)
            bbox_iou = _bbox_iou_xyxy(reference.bbox_xyxy, imported.bbox_xyxy)
            centroid_proximity = _centroid_proximity_score(
                reference.centroid,
                imported.centroid,
                reference.bbox_xyxy,
            )
            score = (0.5 * polygon_iou) + (0.3 * bbox_iou) + (0.2 * centroid_proximity)
            if score <= 0.0:
                continue
            candidates.append(
                _BalloonMatch(
                    reference=reference,
                    imported=imported,
                    polygon_iou=polygon_iou,
                    bbox_iou=bbox_iou,
                    centroid_proximity=centroid_proximity,
                    score=float(score),
                )
            )

    candidates.sort(key=lambda item: item.score, reverse=True)
    matched_reference_ids: set[str] = set()
    matched_import_ids: set[str] = set()
    matches: list[_BalloonMatch] = []
    for candidate in candidates:
        if candidate.reference.balloon_id in matched_reference_ids:
            continue
        if candidate.imported.balloon_id in matched_import_ids:
            continue
        matches.append(candidate)
        matched_reference_ids.add(candidate.reference.balloon_id)
        matched_import_ids.add(candidate.imported.balloon_id)

    unmatched_reference_ids = [
        record.balloon_id for record in reference_records if record.balloon_id not in matched_reference_ids
    ]
    unmatched_import_ids = [
        record.balloon_id for record in import_records if record.balloon_id not in matched_import_ids
    ]
    return matches, unmatched_reference_ids, unmatched_import_ids


def _expanded_crop_window(
    image_shape: tuple[int, int],
    reference_bbox: tuple[int, int, int, int],
    import_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x1 = min(reference_bbox[0], import_bbox[0])
    y1 = min(reference_bbox[1], import_bbox[1])
    x2 = max(reference_bbox[2], import_bbox[2])
    y2 = max(reference_bbox[3], import_bbox[3])
    padding = max(16, int(max(x2 - x1, y2 - y1) * 0.12))
    return (
        max(x1 - padding, 0),
        max(y1 - padding, 0),
        min(x2 + padding, width),
        min(y2 + padding, height),
    )


def _local_bbox_affine_matrix(
    *,
    crop_origin: tuple[int, int],
    reference_bbox: tuple[int, int, int, int],
    import_bbox: tuple[int, int, int, int],
) -> Any:
    _, np = _require_registration_runtime()
    crop_x, crop_y = crop_origin
    ref_x1, ref_y1, ref_x2, ref_y2 = reference_bbox
    imp_x1, imp_y1, imp_x2, imp_y2 = import_bbox
    ref_x1 -= crop_x
    ref_y1 -= crop_y
    ref_x2 -= crop_x
    ref_y2 -= crop_y
    imp_x1 -= crop_x
    imp_y1 -= crop_y
    imp_x2 -= crop_x
    imp_y2 -= crop_y
    imp_width = max(imp_x2 - imp_x1, 1)
    imp_height = max(imp_y2 - imp_y1, 1)
    ref_width = max(ref_x2 - ref_x1, 1)
    ref_height = max(ref_y2 - ref_y1, 1)
    scale_x = ref_width / float(imp_width)
    scale_y = ref_height / float(imp_height)
    translate_x = ref_x1 - (scale_x * imp_x1)
    translate_y = ref_y1 - (scale_y * imp_y1)
    return np.array(
        [[scale_x, 0.0, translate_x], [0.0, scale_y, translate_y]],
        dtype=np.float32,
    )


def _detector_warning_level(
    matched_ratio: float,
    median_balloon_iou: float,
    unmatched_reference_ids: list[str],
) -> str:
    if matched_ratio >= 0.80 and median_balloon_iou >= 0.70 and not unmatched_reference_ids:
        return "normal"
    if matched_ratio >= 0.50 and median_balloon_iou >= 0.45:
        return "warning"
    return "severe"


def refine_aligned_image_with_detector(
    *,
    project_root: Path,
    page_id: str,
    aligned_image: Any,
    registration_report: RegistrationReport,
) -> tuple[Any, RegistrationReport]:
    cv2, np = _require_registration_runtime()
    page_manifest = load_page_manifest(project_root, page_id)
    reference_records = _build_reference_balloon_records(
        page_id,
        page_manifest.balloons,
        aligned_image.shape[:2],
    )
    if not reference_records:
        registration_report.detector_alignment_mode = "reference_missing"
        registration_report.notes.append("detector-based balloon refinement skipped because no reference balloons are stored")
        return aligned_image, registration_report

    detector_name = page_manifest.step1_detector_name or "manga109_seg_v1"
    predictions, resolved_detector_name, detector_error = _run_import_detector(aligned_image, detector_name)
    import_records = _build_import_balloon_records(predictions, aligned_image.shape[:2])
    registration_report.reference_balloon_count = len(reference_records)
    registration_report.import_balloon_count = len(import_records)

    if detector_error is not None:
        registration_report.detector_alignment_mode = "page_level_fallback"
        registration_report.notes.append(detector_error)
        return aligned_image, registration_report
    if not import_records:
        registration_report.detector_alignment_mode = resolved_detector_name
        registration_report.warning_level = "severe"
        registration_report.notes.append("detector-based import refinement found no balloons in the aligned import")
        return aligned_image, registration_report

    matches, unmatched_reference_ids, unmatched_import_ids = _match_detector_balloons(reference_records, import_records)
    registration_report.detector_alignment_mode = resolved_detector_name
    registration_report.matched_balloon_count = len(matches)
    registration_report.unmatched_reference_balloon_ids = unmatched_reference_ids
    registration_report.unmatched_import_balloon_ids = unmatched_import_ids

    if not matches:
        registration_report.matched_ratio = 0.0
        registration_report.median_balloon_iou = 0.0
        registration_report.warning_level = "severe"
        registration_report.notes.append("detector-based balloon refinement found no reference/import matches and kept page-level alignment")
        return aligned_image, registration_report

    refined_image = aligned_image.copy()
    final_ious: list[float] = []
    local_refine_applied_count = 0
    for match in matches:
        coarse_iou = match.polygon_iou
        window = _expanded_crop_window(aligned_image.shape[:2], match.reference.bbox_xyxy, match.imported.bbox_xyxy)
        x1, y1, x2, y2 = window
        crop = aligned_image[y1:y2, x1:x2].copy()
        reference_mask_crop = match.reference.mask[y1:y2, x1:x2]
        import_mask_crop = match.imported.mask[y1:y2, x1:x2]
        local_matrix = _local_bbox_affine_matrix(
            crop_origin=(x1, y1),
            reference_bbox=match.reference.bbox_xyxy,
            import_bbox=match.imported.bbox_xyxy,
        )
        warped_crop = _warp_affine(crop, local_matrix, (x2 - x1, y2 - y1), is_mask=False)
        warped_mask = _warp_affine(import_mask_crop, local_matrix, (x2 - x1, y2 - y1), is_mask=True)
        refined_iou = _mask_iou(reference_mask_crop, warped_mask)

        chosen_crop = crop
        chosen_iou = coarse_iou
        if refined_iou >= (coarse_iou + 0.03):
            chosen_crop = warped_crop
            chosen_iou = refined_iou
            local_refine_applied_count += 1
        else:
            ecc_score, ecc_delta = _try_ecc_refinement(reference_mask_crop, warped_mask)
            if ecc_score is not None and ecc_delta is not None:
                ecc_matrix = _compose_affine(ecc_delta, local_matrix)
                ecc_crop = _warp_affine(crop, ecc_matrix, (x2 - x1, y2 - y1), is_mask=False)
                ecc_mask = _warp_affine(import_mask_crop, ecc_matrix, (x2 - x1, y2 - y1), is_mask=True)
                ecc_iou = _mask_iou(reference_mask_crop, ecc_mask)
                if ecc_iou > max(chosen_iou, refined_iou):
                    chosen_crop = ecc_crop
                    chosen_iou = ecc_iou
                    local_refine_applied_count += 1

        mask_bool = reference_mask_crop > 0
        refined_target = refined_image[y1:y2, x1:x2]
        refined_target[mask_bool] = chosen_crop[mask_bool]
        refined_image[y1:y2, x1:x2] = refined_target
        final_ious.append(chosen_iou)

    if final_ious:
        registration_report.median_balloon_iou = float(np.median(np.array(final_ious, dtype=np.float32)))
    registration_report.matched_ratio = len(matches) / float(max(len(reference_records), 1))
    registration_report.local_refine_applied_count = local_refine_applied_count
    registration_report.final_score = max(registration_report.final_score, registration_report.median_balloon_iou)
    registration_report.warning_level = _detector_warning_level(
        registration_report.matched_ratio,
        registration_report.median_balloon_iou,
        unmatched_reference_ids,
    )
    registration_report.notes.append(
        "detector-based balloon refinement used "
        f"{resolved_detector_name}: matched {registration_report.matched_balloon_count}/"
        f"{registration_report.reference_balloon_count} balloon(s), "
        f"median_iou={registration_report.median_balloon_iou:.4f}"
    )
    if unmatched_reference_ids:
        registration_report.notes.append(
            "unmatched reference balloons: " + ", ".join(unmatched_reference_ids)
        )
    if unmatched_import_ids:
        registration_report.notes.append(
            "unmatched import balloons: " + ", ".join(unmatched_import_ids)
        )
    return refined_image, registration_report


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
