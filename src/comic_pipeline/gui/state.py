from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comic_pipeline.project import (
    create_project,
    load_page_manifest,
    load_project_manifest,
    page_manifest_path,
    project_manifest_path,
    read_json,
    save_project_manifest,
    scan_pages,
)
from comic_pipeline.types import PageManifest, ProjectManifest

STATUS_ORDER = ["pending", "mask_ready", "nano_pending", "restored", "done", "check"]
STATUS_LABELS = {
    "pending": "대기",
    "mask_ready": "검출 완료",
    "nano_pending": "Nano 대기",
    "restored": "복구 완료",
    "done": "완료",
    "check": "검토 필요",
}

PREVIEW_FIELDS = {
    "원본": "original_path",
    "Union Mask": "balloon_union_mask_path",
    "Balloons Only": "balloons_only_rgba_path",
    "Step 3 Raw": "nano_banana_raw_path",
    "Step 4 RGBA": "nano_banana_rgba_path",
    "Final Composite": "final_composite_path",
    "Registration Overlay": "overlay_preview_path",
    "Diff Preview": "diff_preview_path",
}


@dataclass(slots=True)
class PageDetail:
    page_manifest: PageManifest
    step1_report: dict[str, Any] | None
    step2_report: dict[str, Any] | None
    step3_report: dict[str, Any] | None
    step4_report: dict[str, Any] | None
    step5_report: dict[str, Any] | None
    step6_report: dict[str, Any] | None
    registration_report: dict[str, Any] | None


def ensure_project_ready(
    project_root: Path,
    *,
    profile_mode: str = "auto",
    target_language: str = "한국어",
) -> ProjectManifest:
    project_root = project_root.resolve()
    if not project_manifest_path(project_root).exists():
        create_project(project_root, profile_mode=profile_mode, target_language=target_language)
    scan_pages(project_root)
    return load_project_manifest(project_root)


def load_project_state(project_root: Path) -> tuple[ProjectManifest, list[PageManifest]]:
    project_root = project_root.resolve()
    project_manifest = load_project_manifest(project_root)
    page_manifests = [
        load_page_manifest(project_root, path.stem.replace(".page", ""))
        for path in sorted((project_root / "pages").glob("*.page.json"))
    ]
    page_manifests.sort(key=lambda item: (STATUS_ORDER.index(item.status) if item.status in STATUS_ORDER else 99, item.page_id))
    return project_manifest, page_manifests


def save_project_settings(
    project_root: Path,
    *,
    integration_mode: str,
    provider: str,
    model: str,
    target_language: str,
    include_original_page: bool,
) -> ProjectManifest:
    manifest = load_project_manifest(project_root)
    manifest.nano_integration_mode = integration_mode
    manifest.nano_provider = provider
    manifest.nano_model = model
    manifest.nano_target_language = target_language
    manifest.nano_include_original_page = include_original_page
    save_project_manifest(project_root, manifest)
    return manifest


def resolve_project_path(project_root: Path, relative_path: str) -> Path | None:
    if not relative_path:
        return None
    path = project_root / relative_path
    return path if path.exists() else None


def page_preview_paths(project_root: Path, page_manifest: PageManifest) -> dict[str, Path | None]:
    previews: dict[str, Path | None] = {}
    for label, field_name in PREVIEW_FIELDS.items():
        relative_path = getattr(page_manifest, field_name)
        previews[label] = resolve_project_path(project_root, relative_path)
    return previews


def _read_optional_json(project_root: Path, relative_path: str) -> dict[str, Any] | None:
    if not relative_path:
        return None
    path = project_root / relative_path
    if not path.exists():
        return None
    return read_json(path)


def load_page_detail(project_root: Path, page_id: str) -> PageDetail:
    page_manifest = load_page_manifest(project_root, page_id)
    return PageDetail(
        page_manifest=page_manifest,
        step1_report=_read_optional_json(project_root, page_manifest.validation_report_path),
        step2_report=_read_optional_json(project_root, page_manifest.step2_validation_report_path),
        step3_report=_read_optional_json(project_root, page_manifest.step3_validation_report_path),
        step4_report=_read_optional_json(project_root, page_manifest.step4_validation_report_path),
        step5_report=_read_optional_json(project_root, page_manifest.step5_validation_report_path),
        step6_report=_read_optional_json(project_root, page_manifest.step6_validation_report_path),
        registration_report=_read_optional_json(project_root, page_manifest.registration_report_path),
    )


def latest_step6_summary(project_root: Path) -> Path | None:
    candidates = sorted((project_root / "artifacts" / "debug").glob("step6_*_summary.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def latest_step6_payload(project_root: Path) -> dict[str, Any] | None:
    summary_path = latest_step6_summary(project_root)
    if summary_path is None:
        return None
    return read_json(summary_path)


def serialize_payload(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "아직 생성되지 않음"
    return json.dumps(payload, indent=2, ensure_ascii=False)


def page_manifest_exists(project_root: Path, page_id: str) -> bool:
    return page_manifest_path(project_root, page_id).exists()

