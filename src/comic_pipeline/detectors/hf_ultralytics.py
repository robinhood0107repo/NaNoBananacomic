from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from comic_pipeline.detectors.base import BalloonDetector
from comic_pipeline.types import BalloonPrediction, RuntimeDependencyError


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class HFUltralyticsDetectorConfig:
    name: str
    repo_id: str
    filename: str
    confidence: float = 0.20
    device: str = "cuda:0"
    cache_dir: Path | None = None
    class_name_keywords: tuple[str, ...] = ("bubble",)


class HFUltralyticsDetector(BalloonDetector):
    is_production_ready = True

    def __init__(self, config: HFUltralyticsDetectorConfig) -> None:
        self.config = config
        self.name = config.name

    def _require_runtime(self) -> tuple[object, object, object, object]:
        try:
            import cv2  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeDependencyError(
                "The HF Ultralytics detector requires ultralytics, torch, "
                "numpy, and opencv-python. Install the GPU runtime first."
            ) from exc
        return cv2, np, YOLO, torch

    def _download_checkpoint(self) -> Path:
        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeDependencyError(
                "The HF Ultralytics detector requires huggingface_hub. "
                "Install the GPU runtime first."
            ) from exc

        cache_dir = self.config.cache_dir or (_repo_root() / ".model_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        hub_cache_dir = cache_dir / "hub"
        xet_cache_dir = cache_dir / "xet"
        hub_cache_dir.mkdir(parents=True, exist_ok=True)
        xet_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("HF_HUB_CACHE", str(hub_cache_dir))
        os.environ.setdefault("HF_XET_CACHE", str(xet_cache_dir))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
        model_path = hf_hub_download(
            repo_id=self.config.repo_id,
            filename=self.config.filename,
            cache_dir=str(hub_cache_dir),
        )
        return Path(model_path)

    def _matches_class(self, class_name: str, total_class_count: int) -> bool:
        lowered = class_name.lower()
        keywords = tuple(keyword.lower() for keyword in self.config.class_name_keywords)
        if total_class_count <= 1:
            return True
        return any(keyword in lowered for keyword in keywords)

    def predict(self, image: Any) -> list[BalloonPrediction]:
        cv2, np, YOLO, torch = self._require_runtime()
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeDependencyError(
                "CUDA is required for this detector, but the current session cannot access "
                "an NVIDIA device. In this Codex sandbox, /dev/dxg is not exposed. "
                "Run the detector from a normal WSL terminal with GPU access."
            )
        model_path = self._download_checkpoint()
        model = YOLO(str(model_path))
        results = model.predict(
            source=image,
            conf=self.config.confidence,
            verbose=False,
            device=self.config.device,
        )
        if not results:
            return []

        result = results[0]
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = getattr(result, "names", {}) or {}
        class_names = set(names.values()) if isinstance(names, dict) else set()
        predictions: list[BalloonPrediction] = []
        total = len(boxes)
        mask_segments = getattr(masks, "xy", None) if masks is not None else None

        for index in range(total):
            class_id = 0
            if hasattr(boxes, "cls") and len(boxes.cls) > index:
                class_id = int(boxes.cls[index].item())

            if isinstance(names, dict) and names:
                class_name = str(names.get(class_id, class_id))
                if not self._matches_class(class_name, len(class_names)):
                    continue

            xyxy_tensor = boxes.xyxy[index]
            bbox_xyxy = [int(round(float(value))) for value in xyxy_tensor.tolist()]

            polygon: list[list[int]]
            if mask_segments is not None and len(mask_segments) > index and len(mask_segments[index]) >= 3:
                polygon = [
                    [int(round(float(x))), int(round(float(y)))]
                    for x, y in mask_segments[index].tolist()
                ]
            else:
                x1, y1, x2, y2 = bbox_xyxy
                polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            contour = np.array(polygon, dtype=np.int32)
            area = float(cv2.contourArea(contour))
            if area <= 0:
                continue

            confidence = 0.0
            if hasattr(boxes, "conf") and len(boxes.conf) > index:
                confidence = float(boxes.conf[index].item())

            predictions.append(
                BalloonPrediction(
                    bbox_xyxy=bbox_xyxy,
                    polygon=polygon,
                    area=area,
                    confidence=confidence,
                    model_name=self.name,
                )
            )

        predictions.sort(key=lambda item: item.area, reverse=True)
        return predictions
