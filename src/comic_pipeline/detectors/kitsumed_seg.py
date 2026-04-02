from __future__ import annotations

from comic_pipeline.detectors.hf_ultralytics import HFUltralyticsDetector, HFUltralyticsDetectorConfig


class KitsumedSegmentationDetector(HFUltralyticsDetector):
    def __init__(self) -> None:
        super().__init__(
            HFUltralyticsDetectorConfig(
                name="kitsumed_seg_v1",
                repo_id="kitsumed/yolov8m_seg-speech-bubble",
                filename="model.pt",
                confidence=0.20,
                class_name_keywords=("bubble",),
            )
        )
