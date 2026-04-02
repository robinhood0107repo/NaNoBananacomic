from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from comic_pipeline.types import BalloonPrediction


class BalloonDetector(ABC):
    name: str

    @abstractmethod
    def predict(self, image: Any) -> list[BalloonPrediction]:
        raise NotImplementedError

