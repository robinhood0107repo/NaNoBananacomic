from __future__ import annotations

from comic_pipeline.detectors.hf_ultralytics import HFUltralyticsDetector, HFUltralyticsDetectorConfig


class OgkaluBubbleDetector(HFUltralyticsDetector):
    def __init__(self) -> None:
        super().__init__(
            HFUltralyticsDetectorConfig(
                name="ogkalu_bubble_v1",
                repo_id="ogkalu/comic-speech-bubble-detector-yolov8m",
                filename="comic-speech-bubble-detector.pt",
                confidence=0.20,
                class_name_keywords=("bubble",),
            )
        )
