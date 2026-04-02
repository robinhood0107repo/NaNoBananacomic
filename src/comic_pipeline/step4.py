from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from comic_pipeline.make_balloons_only import build_soft_alpha_from_union_mask
from comic_pipeline.project import (
    load_page_manifest,
    load_project_manifest,
    save_page_manifest,
    save_project_manifest,
    write_json,
)
from comic_pipeline.types import RuntimeDependencyError, Step4ValidationReport


class Step4SourceError(RuntimeError):
    """Raised when Phase 4 cannot resolve or read a usable source image."""

    def __init__(self, message: str, *, readable_image: bool) -> None:
        super().__init__(message)
        self.readable_image = readable_image


def _require_step4_runtime() -> tuple[object, object]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeDependencyError(
            "Step 4 runtime requires numpy and opencv-python. "
            "Install the dependencies from pyproject.toml before running Step 4 commands."
        ) from exc
    return cv2, np


def _as_relative(project_root: Path, path: Path) -> str:
    return str(path.relative_to(project_root))


def _project_timestamp(project_root: Path) -> None:
    project_manifest = load_project_manifest(project_root)
    project_manifest.last_run_id = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    save_project_manifest(project_root, project_manifest)


def _load_step4_context(project_root: Path, page_id: str) -> dict[str, Any]:
    cv2, _ = _require_step4_runtime()
    project_manifest = load_project_manifest(project_root)
    page_manifest = load_page_manifest(project_root, page_id)

    if not page_manifest.balloon_union_mask_path:
        raise FileNotFoundError(
            f"Page {page_id} has no saved union mask. Run Step 1 first."
        )

    original_path = project_root / page_manifest.original_path
    union_mask_path = project_root / page_manifest.balloon_union_mask_path
    if not original_path.exists():
        raise FileNotFoundError(f"Original page does not exist: {original_path}")
    if not union_mask_path.exists():
        raise FileNotFoundError(f"Union mask does not exist: {union_mask_path}")

    original_image = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
    union_mask = cv2.imread(str(union_mask_path), cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise FileNotFoundError(f"Unable to read original page: {original_path}")
    if union_mask is None:
        raise FileNotFoundError(f"Unable to read union mask: {union_mask_path}")

    expected_height, expected_width = original_image.shape[:2]
    if page_manifest.width <= 0 or page_manifest.height <= 0:
        page_manifest.width = expected_width
        page_manifest.height = expected_height
        save_page_manifest(project_root, page_manifest)

    return {
        "project_manifest": project_manifest,
        "page_manifest": page_manifest,
        "expected_size": (page_manifest.width, page_manifest.height),
        "union_mask": union_mask,
        "original_path": original_path,
    }


def _discover_manual_import_candidate(project_root: Path, page_id: str) -> Path | None:
    project_manifest = load_project_manifest(project_root)
    imports_dir = project_root / project_manifest.imports_dir / "nano"
    if not imports_dir.exists():
        return None

    candidates: list[Path] = []
    for path in sorted(imports_dir.iterdir()):
        if not path.is_file():
            continue
        stem = path.stem
        if stem == page_id or stem.startswith(f"{page_id}_"):
            candidates.append(path)
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, float]:
        stem = path.stem
        if stem == f"{page_id}_submitted":
            priority = 0
        elif stem == f"{page_id}_raw":
            priority = 1
        else:
            priority = 2
        return (priority, -path.stat().st_mtime)

    return sorted(candidates, key=sort_key)[0]


def _read_image(path: Path) -> Any:
    cv2, _ = _require_step4_runtime()
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise Step4SourceError(f"Unable to read Phase 4 source image: {path}", readable_image=False)
    return image


def _resolve_step4_source(project_root: Path, page_id: str) -> tuple[Path, str]:
    page_manifest = load_page_manifest(project_root, page_id)

    manual_path: Path | None = None
    if page_manifest.nano_source_kind == "manual_web":
        if page_manifest.nano_manual_import_path:
            candidate = project_root / page_manifest.nano_manual_import_path
            if candidate.exists():
                manual_path = candidate
        if manual_path is None:
            manual_path = _discover_manual_import_candidate(project_root, page_id)
        if manual_path is not None:
            page_manifest.nano_manual_import_path = _as_relative(project_root, manual_path)
            save_page_manifest(project_root, page_manifest)
            return manual_path, "manual_web"

    if page_manifest.nano_banana_raw_path:
        raw_path = project_root / page_manifest.nano_banana_raw_path
        if raw_path.exists():
            return raw_path, page_manifest.nano_source_kind or "raw_contract"

    if page_manifest.nano_manual_import_path:
        candidate = project_root / page_manifest.nano_manual_import_path
        if candidate.exists():
            manual_path = candidate
    if manual_path is None:
        manual_path = _discover_manual_import_candidate(project_root, page_id)

    if manual_path is None:
        raise Step4SourceError(
            "No Phase 4 input was found. Import an external result first or place it in imports/nano/.",
            readable_image=False,
        )

    page_manifest.nano_manual_import_path = _as_relative(project_root, manual_path)
    if not page_manifest.nano_source_kind:
        page_manifest.nano_source_kind = "manual_web"
    save_page_manifest(project_root, page_manifest)
    return manual_path, page_manifest.nano_source_kind or "manual_web"


def _to_rgb(image: Any) -> Any:
    cv2, _ = _require_step4_runtime()
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image[:, :, :3]


def _normalize_source_to_raw_contract(
    *,
    source_path: Path,
    raw_output_path: Path,
    expected_size: tuple[int, int],
) -> tuple[bool, bool]:
    cv2, _ = _require_step4_runtime()
    source_image = _read_image(source_path)
    source_height, source_width = source_image.shape[:2]
    source_size_matches = (source_width, source_height) == expected_size
    alignment_applied = False

    normalized = source_image
    if not source_size_matches:
        interpolation = (
            cv2.INTER_AREA
            if source_width > expected_size[0] or source_height > expected_size[1]
            else cv2.INTER_LINEAR
        )
        normalized = cv2.resize(source_image, expected_size, interpolation=interpolation)
        alignment_applied = True

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(raw_output_path), normalized):
        raise OSError(f"Unable to write normalized Step 4 raw image: {raw_output_path}")
    return source_size_matches, alignment_applied


def _compose_restored_rgba(raw_image: Any, restored_alpha: Any) -> Any:
    _, np = _require_step4_runtime()
    raw_rgb = _to_rgb(raw_image)
    restored = np.zeros((raw_rgb.shape[0], raw_rgb.shape[1], 4), dtype=np.uint8)
    active_pixels = restored_alpha > 0
    restored[:, :, 3] = restored_alpha
    restored[active_pixels, :3] = raw_rgb[active_pixels]
    return restored


def summarize_step4_validation(
    *,
    page_id: str,
    source_kind: str | None,
    readable_image: bool,
    source_size_matches: bool,
    alignment_applied: bool,
    restored_size_matches: bool,
    has_alpha_channel: bool,
    outside_alpha_sum: int,
    outside_rgb_nonzero_pixels: int,
    inside_alpha_preservation_ratio: float,
    nonzero_alpha_pixels: int,
    min_inside_alpha_preservation_ratio: float = 0.90,
) -> Step4ValidationReport:
    notes: list[str] = []
    if not readable_image:
        notes.append("step4 source image is unreadable")
    if not source_size_matches and alignment_applied:
        notes.append("source image size did not match the original page size and was aligned locally")
    elif not source_size_matches:
        notes.append("source image size does not match the original page size")
    if not restored_size_matches:
        notes.append("restored RGBA size does not match the original page size")
    if not has_alpha_channel:
        notes.append("restored RGBA output is missing an alpha channel")
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
        notes.append("restored RGBA output has no nonzero alpha pixels")

    passed = (
        readable_image
        and restored_size_matches
        and has_alpha_channel
        and outside_alpha_sum == 0
        and outside_rgb_nonzero_pixels == 0
        and inside_alpha_preservation_ratio >= min_inside_alpha_preservation_ratio
        and nonzero_alpha_pixels > 0
    )
    if passed:
        notes.append("step4 validation passed and the translated balloon layer is ready for final composite")

    return Step4ValidationReport(
        page_id=page_id,
        source_kind=source_kind,
        readable_image=readable_image,
        source_size_matches=source_size_matches,
        alignment_applied=alignment_applied,
        restored_size_matches=restored_size_matches,
        has_alpha_channel=has_alpha_channel,
        outside_alpha_sum=outside_alpha_sum,
        outside_rgb_nonzero_pixels=outside_rgb_nonzero_pixels,
        inside_alpha_preservation_ratio=inside_alpha_preservation_ratio,
        nonzero_alpha_pixels=nonzero_alpha_pixels,
        passed=passed,
        notes=notes,
    )


def _save_step4_report(project_root: Path, page_id: str, report: Step4ValidationReport) -> Step4ValidationReport:
    report_path = project_root / "artifacts" / "debug" / f"{page_id}_step4_validation.json"
    write_json(report_path, report.to_dict())
    page_manifest = load_page_manifest(project_root, page_id)
    page_manifest.step4_validation_report_path = _as_relative(project_root, report_path)
    page_manifest.status = "restored" if report.passed else "check"
    save_page_manifest(project_root, page_manifest)
    return report


def _save_step4_failure_report(
    project_root: Path,
    page_id: str,
    *,
    source_kind: str | None,
    readable_image: bool,
    notes: list[str],
) -> Step4ValidationReport:
    report = Step4ValidationReport(
        page_id=page_id,
        source_kind=source_kind,
        readable_image=readable_image,
        source_size_matches=False,
        alignment_applied=False,
        restored_size_matches=False,
        has_alpha_channel=False,
        outside_alpha_sum=0,
        outside_rgb_nonzero_pixels=0,
        inside_alpha_preservation_ratio=0.0,
        nonzero_alpha_pixels=0,
        passed=False,
        notes=notes,
    )
    return _save_step4_report(project_root, page_id, report)


def _source_size_metadata(
    project_root: Path,
    page_id: str,
    *,
    expected_size: tuple[int, int],
    source_kind: str | None,
) -> tuple[bool, bool, bool]:
    page_manifest = load_page_manifest(project_root, page_id)
    readable_image = True
    source_size_matches = True
    alignment_applied = False

    if source_kind == "manual_web" and page_manifest.nano_manual_import_path:
        manual_path = project_root / page_manifest.nano_manual_import_path
        if not manual_path.exists():
            return (False, False, False)
        manual_image = _read_image(manual_path)
        source_height, source_width = manual_image.shape[:2]
        source_size_matches = (source_width, source_height) == expected_size
        raw_path = project_root / page_manifest.nano_banana_raw_path if page_manifest.nano_banana_raw_path else None
        alignment_applied = (not source_size_matches) and raw_path is not None and raw_path.exists()
        return (readable_image, source_size_matches, alignment_applied)

    if page_manifest.nano_banana_raw_path:
        raw_path = project_root / page_manifest.nano_banana_raw_path
        if not raw_path.exists():
            return (False, False, False)
        raw_image = _read_image(raw_path)
        source_height, source_width = raw_image.shape[:2]
        source_size_matches = (source_width, source_height) == expected_size
        return (readable_image, source_size_matches, False)

    return (False, False, False)


def validate_step4(project_root: Path, page_id: str) -> Step4ValidationReport:
    cv2, np = _require_step4_runtime()
    context = _load_step4_context(project_root, page_id)
    page_manifest = context["page_manifest"]
    if not page_manifest.nano_banana_rgba_path:
        raise FileNotFoundError(
            f"Page {page_id} has no Step 4 output yet. Run restore-alpha first."
        )

    rgba_path = project_root / page_manifest.nano_banana_rgba_path
    if not rgba_path.exists():
        raise FileNotFoundError(f"Unable to read Step 4 RGBA result: {rgba_path}")

    rgba_image = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if rgba_image is None:
        return _save_step4_failure_report(
            project_root,
            page_id,
            source_kind=page_manifest.nano_source_kind,
            readable_image=False,
            notes=["restored RGBA output is unreadable"],
        )

    readable_image, source_size_matches, alignment_applied = _source_size_metadata(
        project_root,
        page_id,
        expected_size=context["expected_size"],
        source_kind=page_manifest.nano_source_kind,
    )
    restored_size_matches = (rgba_image.shape[1], rgba_image.shape[0]) == context["expected_size"]
    has_alpha_channel = rgba_image.ndim == 3 and rgba_image.shape[2] == 4
    outside_alpha_sum = 0
    outside_rgb_nonzero_pixels = 0
    inside_alpha_preservation_ratio = 0.0
    nonzero_alpha_pixels = 0

    if restored_size_matches and has_alpha_channel:
        _, support = build_soft_alpha_from_union_mask(context["union_mask"])
        outside_support = support == 0
        alpha = rgba_image[:, :, 3]
        rgb = rgba_image[:, :, :3]
        outside_alpha_sum = int(alpha[outside_support].sum())
        outside_rgb = rgb[outside_support]
        if len(outside_rgb) > 0:
            outside_rgb_nonzero_pixels = int(
                np.count_nonzero(np.any(outside_rgb > 0, axis=1))
            )

        inside_original = context["union_mask"] > 0
        inside_original_pixels = int(np.count_nonzero(inside_original))
        if inside_original_pixels > 0:
            inside_alpha_preservation_ratio = float(
                np.count_nonzero(alpha[inside_original] > 0) / inside_original_pixels
            )
        nonzero_alpha_pixels = int(np.count_nonzero(alpha > 0))

    report = summarize_step4_validation(
        page_id=page_id,
        source_kind=page_manifest.nano_source_kind,
        readable_image=readable_image,
        source_size_matches=source_size_matches,
        alignment_applied=alignment_applied,
        restored_size_matches=restored_size_matches,
        has_alpha_channel=has_alpha_channel,
        outside_alpha_sum=outside_alpha_sum,
        outside_rgb_nonzero_pixels=outside_rgb_nonzero_pixels,
        inside_alpha_preservation_ratio=inside_alpha_preservation_ratio,
        nonzero_alpha_pixels=nonzero_alpha_pixels,
    )
    return _save_step4_report(project_root, page_id, report)


def restore_alpha(project_root: Path, page_id: str) -> dict[str, Any]:
    cv2, _ = _require_step4_runtime()
    context = _load_step4_context(project_root, page_id)
    page_manifest = context["page_manifest"]

    try:
        source_path, source_kind = _resolve_step4_source(project_root, page_id)
        raw_output_path = project_root / "artifacts" / "nano" / f"{page_id}_raw.png"
        source_size_matches, alignment_applied = _normalize_source_to_raw_contract(
            source_path=source_path,
            raw_output_path=raw_output_path,
            expected_size=context["expected_size"],
        )
        raw_image = cv2.imread(str(raw_output_path), cv2.IMREAD_UNCHANGED)
        if raw_image is None:
            raise Step4SourceError(
                f"Unable to read normalized Step 4 raw image: {raw_output_path}",
                readable_image=False,
            )

        restored_alpha, _ = build_soft_alpha_from_union_mask(context["union_mask"])
        restored_rgba = _compose_restored_rgba(raw_image, restored_alpha)
        rgba_output_path = project_root / "artifacts" / "nano" / f"{page_id}_rgba.png"
        rgba_output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(rgba_output_path), restored_rgba):
            raise OSError(f"Unable to write Step 4 RGBA result: {rgba_output_path}")

        page_manifest = load_page_manifest(project_root, page_id)
        page_manifest.nano_source_kind = source_kind
        page_manifest.nano_banana_raw_path = _as_relative(project_root, raw_output_path)
        page_manifest.nano_banana_rgba_path = _as_relative(project_root, rgba_output_path)
        save_page_manifest(project_root, page_manifest)

        report = validate_step4(project_root, page_id)
        if not source_size_matches and alignment_applied and report.passed:
            report.notes.append(
                "manual import size mismatch was tolerated for Phase 4 by aligning to the page canvas"
            )
            _save_step4_report(project_root, page_id, report)
    except (FileNotFoundError, OSError, Step4SourceError) as exc:
        page_manifest.nano_banana_rgba_path = ""
        save_page_manifest(project_root, page_manifest)
        report = _save_step4_failure_report(
            project_root,
            page_id,
            source_kind=page_manifest.nano_source_kind,
            readable_image=getattr(exc, "readable_image", False),
            notes=[str(exc)],
        )

    _project_timestamp(project_root)
    final_page_manifest = load_page_manifest(project_root, page_id)
    return {
        "page_id": page_id,
        "nano_manual_import_path": final_page_manifest.nano_manual_import_path,
        "nano_banana_raw_path": final_page_manifest.nano_banana_raw_path,
        "nano_banana_rgba_path": final_page_manifest.nano_banana_rgba_path,
        "step4_validation_report_path": final_page_manifest.step4_validation_report_path,
        "status": final_page_manifest.status,
        "passed": report.passed,
    }


def format_step4_report(report: Step4ValidationReport) -> str:
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
