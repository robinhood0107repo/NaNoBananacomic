from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from comic_pipeline.detectors.base import BalloonDetector
from comic_pipeline.types import BalloonPrediction, RuntimeDependencyError


@dataclass(slots=True)
class ContourDetectorConfig:
    bright_threshold: int = 185
    min_component_area: int = 800
    min_width: int = 24
    min_height: int = 24
    max_aspect_ratio: float = 6.0
    min_fill_ratio: float = 0.12
    max_fill_ratio: float = 0.95


class ContourBalloonDetector(BalloonDetector):
    name = "contour_baseline_v1"
    is_production_ready = False

    def __init__(self, config: ContourDetectorConfig | None = None) -> None:
        self.config = config or ContourDetectorConfig()

    def predict(self, image: Any) -> list[BalloonPrediction]:
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeDependencyError(
                "Step 1 detection requires numpy and opencv-python. "
                "Install the dependencies from pyproject.toml first."
            ) from exc

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred,
            self.config.bright_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        # Remove border-connected bright background so enclosed bright balloons remain.
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
        closed = cv2.morphologyEx(flood, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        predictions: list[BalloonPrediction] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.min_component_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width < self.config.min_width or height < self.config.min_height:
                continue

            aspect_ratio = max(width / max(height, 1), height / max(width, 1))
            if aspect_ratio > self.config.max_aspect_ratio:
                continue

            fill_ratio = area / float(width * height)
            if fill_ratio < self.config.min_fill_ratio or fill_ratio > self.config.max_fill_ratio:
                continue

            component_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
            mean_brightness = cv2.mean(gray, mask=component_mask)[0] / 255.0
            if mean_brightness < 0.55:
                continue

            flattened = contour.reshape(-1, 2)
            polygon = [[int(px), int(py)] for px, py in flattened.tolist()]
            predictions.append(
                BalloonPrediction(
                    bbox_xyxy=[int(x), int(y), int(x + width), int(y + height)],
                    polygon=polygon,
                    area=area,
                    confidence=max(0.05, min(0.99, mean_brightness * fill_ratio)),
                    model_name=self.name,
                )
            )

        predictions.sort(key=lambda item: item.area, reverse=True)
        return predictions
