from comic_pipeline.detectors.base import BalloonDetector
from comic_pipeline.detectors.contour import ContourBalloonDetector
from comic_pipeline.detectors.kitsumed_seg import KitsumedSegmentationDetector
from comic_pipeline.detectors.manga109_seg import Manga109SegmentationDetector
from comic_pipeline.detectors.ogkalu_bubble import OgkaluBubbleDetector

__all__ = [
    "BalloonDetector",
    "ContourBalloonDetector",
    "KitsumedSegmentationDetector",
    "Manga109SegmentationDetector",
    "OgkaluBubbleDetector",
]
