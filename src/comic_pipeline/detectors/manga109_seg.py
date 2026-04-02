from __future__ import annotations

from comic_pipeline.detectors.hf_ultralytics import HFUltralyticsDetector, HFUltralyticsDetectorConfig


class Manga109SegmentationDetector(HFUltralyticsDetector):
    def __init__(self) -> None:
        super().__init__(
            HFUltralyticsDetectorConfig(
                name="manga109_seg_v1",
                repo_id="huyvux3005/manga109-segmentation-bubble",
                filename="best.pt",
                confidence=0.20,
                class_name_keywords=("bubble",),
            )
        )
