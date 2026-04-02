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
    nano_banana_raw_path: str = ""
    nano_banana_rgba_path: str = ""
    final_composite_path: str = ""
    result_path: str = ""
    overlay_preview_path: str = ""
    validation_report_path: str = ""
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
    passed: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

