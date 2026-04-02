from __future__ import annotations

from typing import Callable

from comic_pipeline.detectors.base import BalloonDetector
from comic_pipeline.detectors.contour import ContourBalloonDetector
from comic_pipeline.detectors.kitsumed_seg import KitsumedSegmentationDetector
from comic_pipeline.detectors.manga109_seg import Manga109SegmentationDetector
from comic_pipeline.detectors.ogkalu_bubble import OgkaluBubbleDetector

DetectorFactory = Callable[[], BalloonDetector]

DETECTOR_FACTORIES: dict[str, DetectorFactory] = {
    "manga109_seg_v1": Manga109SegmentationDetector,
    "kitsumed_seg_v1": KitsumedSegmentationDetector,
    "ogkalu_bubble_v1": OgkaluBubbleDetector,
    "contour_baseline_v1": ContourBalloonDetector,
}

DETECTOR_ALIASES = {
    "contour": "contour_baseline_v1",
    "manga109": "manga109_seg_v1",
    "manga109_seg": "manga109_seg_v1",
    "kitsumed": "kitsumed_seg_v1",
    "ogkalu": "ogkalu_bubble_v1",
}

PUBLIC_DETECTOR_NAMES = [
    "manga109_seg_v1",
    "kitsumed_seg_v1",
    "ogkalu_bubble_v1",
]


def normalize_detector_name(name: str) -> str:
    lowered = name.strip().lower()
    canonical = DETECTOR_ALIASES.get(lowered, lowered)
    if canonical not in DETECTOR_FACTORIES:
        available = ", ".join(sorted(DETECTOR_FACTORIES))
        raise ValueError(f"Unsupported detector '{name}'. Available: {available}")
    return canonical


def build_detector(name: str) -> BalloonDetector:
    canonical = normalize_detector_name(name)
    return DETECTOR_FACTORIES[canonical]()


def list_detector_names(include_internal: bool = False) -> list[str]:
    if include_internal:
        return sorted(DETECTOR_FACTORIES)
    return sorted(PUBLIC_DETECTOR_NAMES)
