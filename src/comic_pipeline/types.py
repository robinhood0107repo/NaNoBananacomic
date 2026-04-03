from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class RuntimeDependencyError(RuntimeError):
    """Raised when optional runtime dependencies are missing."""


@dataclass(slots=True)
class BalloonPrediction:
    bbox_xyxy: list[int]
    polygon: list[list[int]]
    area: float
    confidence: float
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProjectManifest:
    project_id: str
    project_root: str
    profile_mode: str = "auto"
    resolved_profile: str = "unknown"
    nano_integration_mode: str = "manual_web"
    nano_provider: str = "gemini"
    nano_model: str = "nano-banana-2"
    nano_target_language: str = "한국어"
    imports_dir: str = "imports"
    nano_include_original_page: bool = True
    page_count: int = 0
    result_dir: str = "result"
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs"
    last_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PageManifest:
    page_id: str
    original_path: str
    width: int = 0
    height: int = 0
    balloon_count: int = 0
    profile: str = "unknown"
    status: str = "pending"
    balloon_union_mask_path: str = ""
    balloons_only_rgba_path: str = ""
    nano_request_dir: str = ""
    nano_request_prompt_path: str = ""
    nano_request_manifest_path: str = ""
    nano_manual_import_path: str = ""
    nano_source_kind: str | None = None
    nano_banana_raw_path: str = ""
    nano_banana_rgba_path: str = ""
    registration_report_path: str = ""
    final_composite_path: str = ""
    result_path: str = ""
    diff_preview_path: str = ""
    overlay_preview_path: str = ""
    validation_report_path: str = ""
    step2_validation_report_path: str = ""
    step3_validation_report_path: str = ""
    step4_validation_report_path: str = ""
    step5_validation_report_path: str = ""
    step6_validation_report_path: str = ""
    step4_source_mode: str = ""
    step4_checkerboard_cleanup_applied: bool = False
    step4_checkerboard_cleaned_pixels: int = 0
    step4_checkerboard_boundary_restored_pixels: int = 0
    step1_detector_name: str = ""
    balloons: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Step1ValidationReport:
    page_id: str
    mask_non_empty: bool
    area_ratio: float
    prediction_count: int
    bbox_out_of_bounds: bool
    area_ratio_in_expected_range: bool
    passed: bool = False
    notes: list[str] = field(default_factory=list)
    detector_names: list[str] = field(default_factory=list)
    production_detector_active: bool = False
    largest_prediction_area_ratio: float = 0.0
    dominant_prediction_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Step2ValidationReport:
    page_id: str
    size_matches: bool
    has_alpha_channel: bool
    outside_alpha_sum: int
    inside_alpha_preservation_ratio: float
    nonzero_alpha_pixels: int
    outside_rgb_nonzero_pixels: int
    passed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Step3ValidationReport:
    page_id: str
    source_kind: str | None
    readable_image: bool
    size_matches: bool
    has_alpha_channel: bool
    opaque_output: bool
    outside_support_rgb_nonzero_pixels: int
    outside_support_alpha_nonzero_pixels: int
    passed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Step4ValidationReport:
    page_id: str
    source_kind: str | None
    source_mode: str
    readable_image: bool
    source_size_matches: bool
    alignment_applied: bool
    restored_size_matches: bool
    has_alpha_channel: bool
    outside_alpha_sum: int
    outside_rgb_nonzero_pixels: int
    inside_alpha_preservation_ratio: float
    nonzero_alpha_pixels: int
    checkerboard_cleanup_applied: bool = False
    checkerboard_cleaned_pixels: int = 0
    checkerboard_boundary_restored_pixels: int = 0
    passed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RegistrationReport:
    page_id: str
    source_kind: str | None
    reference_source: str
    moving_support_source: str
    source_size_matches: bool
    alignment_applied: bool
    method_used: str
    initial_score: float
    refined_score: float
    final_score: float
    warning_level: str
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    transform_matrix: list[list[float]]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Step5ValidationReport:
    page_id: str
    final_exists: bool
    final_size_matches: bool
    overlay_exists: bool
    overlay_size_matches: bool
    registration_report_exists: bool
    registration_score: float
    registration_warning_level: str
    outside_mask_diff_pixels: int
    outside_mask_region_pixels: int
    outside_mask_diff_ratio: float
    passed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Step6ValidationReport:
    scope: str
    page_ids: list[str]
    done_pages: list[str]
    check_pages: list[str]
    missing_artifact_pages: list[str]
    severe_registration_pages: list[str]
    outside_mask_diff_pages: list[str]
    page_summaries: list[dict[str, Any]]
    passed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
