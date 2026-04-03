from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from comic_pipeline.project import load_page_manifest, read_json, save_page_manifest, write_json
from comic_pipeline.types import Step6ValidationReport


def _as_relative(project_root: Path, path: Path) -> str:
    return str(path.relative_to(project_root))


def _load_optional_json(project_root: Path, relative_path: str) -> dict[str, Any] | None:
    if not relative_path:
        return None
    path = project_root / relative_path
    if not path.exists():
        return None
    return read_json(path)


def _collect_page_summary(project_root: Path, page_id: str) -> dict[str, Any]:
    page_manifest = load_page_manifest(project_root, page_id)
    missing_artifacts: list[str] = []

    step1_report = _load_optional_json(project_root, page_manifest.validation_report_path)
    step2_report = _load_optional_json(project_root, page_manifest.step2_validation_report_path)
    step3_report = _load_optional_json(project_root, page_manifest.step3_validation_report_path)
    step4_report = _load_optional_json(project_root, page_manifest.step4_validation_report_path)
    step5_report = _load_optional_json(project_root, page_manifest.step5_validation_report_path)
    registration_report = _load_optional_json(project_root, page_manifest.registration_report_path)
    raw_exists = bool(page_manifest.nano_banana_raw_path and (project_root / page_manifest.nano_banana_raw_path).exists())
    manual_step3_passthrough = (
        page_manifest.nano_source_kind == "manual_web"
        and bool(page_manifest.nano_manual_import_path)
        and raw_exists
        and bool(step4_report and step4_report.get("passed"))
    )

    if step1_report is None:
        missing_artifacts.append("step1_validation_report")
    if step2_report is None:
        missing_artifacts.append("step2_validation_report")
    if not raw_exists:
        missing_artifacts.append("step3_raw")
    if step3_report is None and not manual_step3_passthrough:
        missing_artifacts.append("step3_validation_report")
    if not page_manifest.nano_banana_rgba_path or not (project_root / page_manifest.nano_banana_rgba_path).exists():
        missing_artifacts.append("step4_rgba")
    if step4_report is None:
        missing_artifacts.append("step4_validation_report")
    if registration_report is None:
        missing_artifacts.append("registration_report")
    if not page_manifest.final_composite_path or not (project_root / page_manifest.final_composite_path).exists():
        missing_artifacts.append("final_composite")
    if step5_report is None:
        missing_artifacts.append("step5_validation_report")

    step5_diff_ratio = float(step5_report.get("outside_mask_diff_ratio", 0.0)) if step5_report else 0.0
    registration_warning_level = str(registration_report.get("warning_level", "missing")) if registration_report else "missing"
    registration_score = float(registration_report.get("final_score", 0.0)) if registration_report else 0.0
    detector_alignment_mode = str(registration_report.get("detector_alignment_mode", "")) if registration_report else ""
    matched_ratio = float(registration_report.get("matched_ratio", 0.0)) if registration_report else 0.0
    median_balloon_iou = float(registration_report.get("median_balloon_iou", 0.0)) if registration_report else 0.0
    unmatched_reference_balloon_ids = list(
        registration_report.get("unmatched_reference_balloon_ids", [])
    ) if registration_report else []
    matched_balloon_count = int(registration_report.get("matched_balloon_count", 0)) if registration_report else 0
    reference_balloon_count = int(registration_report.get("reference_balloon_count", 0)) if registration_report else 0
    step3_passed = bool(step3_report and step3_report.get("passed")) or manual_step3_passthrough

    passed = (
        not missing_artifacts
        and bool(step1_report and step1_report.get("passed"))
        and bool(step2_report and step2_report.get("passed"))
        and step3_passed
        and bool(step4_report and step4_report.get("passed"))
        and bool(step5_report and step5_report.get("passed"))
    )

    return {
        "page_id": page_id,
        "status": "done" if passed else "check",
        "manifest_status": page_manifest.status,
        "missing_artifacts": missing_artifacts,
        "registration_warning_level": registration_warning_level,
        "registration_score": registration_score,
        "detector_alignment_mode": detector_alignment_mode,
        "matched_ratio": matched_ratio,
        "median_balloon_iou": median_balloon_iou,
        "matched_balloon_count": matched_balloon_count,
        "reference_balloon_count": reference_balloon_count,
        "unmatched_reference_balloon_ids": unmatched_reference_balloon_ids,
        "outside_mask_diff_ratio": step5_diff_ratio,
        "step1_passed": bool(step1_report and step1_report.get("passed")),
        "step2_passed": bool(step2_report and step2_report.get("passed")),
        "step3_passed": step3_passed,
        "step4_passed": bool(step4_report and step4_report.get("passed")),
        "step5_passed": bool(step5_report and step5_report.get("passed")),
    }


def _discover_page_ids(project_root: Path) -> list[str]:
    return [
        path.name.replace(".page.json", "")
        for path in sorted((project_root / "pages").glob("*.page.json"))
    ]


def validate_step6(
    project_root: Path,
    *,
    page_id: str | None = None,
    validate_all: bool = False,
) -> Step6ValidationReport:
    if validate_all:
        page_ids = _discover_page_ids(project_root)
    elif page_id is not None:
        page_ids = [page_id]
    else:
        raise ValueError("validate_step6 requires --page-id or --all")

    page_summaries = [_collect_page_summary(project_root, target_page_id) for target_page_id in page_ids]
    done_pages = [item["page_id"] for item in page_summaries if item["status"] == "done"]
    check_pages = [item["page_id"] for item in page_summaries if item["status"] == "check"]
    missing_artifact_pages = [item["page_id"] for item in page_summaries if item["missing_artifacts"]]
    severe_registration_pages = [
        item["page_id"]
        for item in page_summaries
        if item["registration_warning_level"] == "severe"
    ]
    outside_mask_diff_pages = [
        item["page_id"]
        for item in page_summaries
        if item["outside_mask_diff_ratio"] > 0.005
    ]
    low_balloon_match_pages = [
        item["page_id"]
        for item in page_summaries
        if item["detector_alignment_mode"]
        and (
            item["matched_ratio"] < 0.80
            or item["median_balloon_iou"] < 0.70
        )
    ]
    unmatched_balloon_pages = [
        item["page_id"]
        for item in page_summaries
        if item["unmatched_reference_balloon_ids"]
    ]
    balloon_match_summary = [
        {
            "page_id": item["page_id"],
            "mode": item["detector_alignment_mode"],
            "matched_balloon_count": item["matched_balloon_count"],
            "reference_balloon_count": item["reference_balloon_count"],
            "matched_ratio": item["matched_ratio"],
            "median_balloon_iou": item["median_balloon_iou"],
            "unmatched_reference_balloon_ids": item["unmatched_reference_balloon_ids"],
        }
        for item in page_summaries
        if item["detector_alignment_mode"]
    ]
    notes: list[str] = []
    if missing_artifact_pages:
        notes.append(f"missing artifacts were detected for {len(missing_artifact_pages)} page(s)")
    if severe_registration_pages:
        notes.append(f"severe registration warnings were detected for {len(severe_registration_pages)} page(s)")
    if outside_mask_diff_pages:
        notes.append(f"outside-mask diff failures were detected for {len(outside_mask_diff_pages)} page(s)")
    if low_balloon_match_pages:
        notes.append(f"low balloon match coverage was detected for {len(low_balloon_match_pages)} page(s)")
    if unmatched_balloon_pages:
        notes.append(f"unmatched reference balloons remain for {len(unmatched_balloon_pages)} page(s)")

    report = Step6ValidationReport(
        scope="all" if validate_all else "page",
        page_ids=page_ids,
        done_pages=done_pages,
        check_pages=check_pages,
        missing_artifact_pages=missing_artifact_pages,
        severe_registration_pages=severe_registration_pages,
        outside_mask_diff_pages=outside_mask_diff_pages,
        page_summaries=page_summaries,
        low_balloon_match_pages=low_balloon_match_pages,
        unmatched_balloon_pages=unmatched_balloon_pages,
        balloon_match_summary=balloon_match_summary,
        passed=(not check_pages),
        notes=notes,
    )
    timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    report_path = project_root / "artifacts" / "debug" / f"step6_{timestamp}_summary.json"
    write_json(report_path, report.to_dict())

    if len(page_ids) == 1:
        page_manifest = load_page_manifest(project_root, page_ids[0])
        page_manifest.step6_validation_report_path = _as_relative(project_root, report_path)
        save_page_manifest(project_root, page_manifest)

    return report


def format_step6_report(report: Step6ValidationReport) -> str:
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
