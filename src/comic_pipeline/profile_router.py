from __future__ import annotations

from pathlib import Path


def route_page_profile(image_path: Path) -> str:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return "unknown"

    image = cv2.imread(str(image_path))
    if image is None:
        return "unknown"

    b_channel, g_channel, r_channel = cv2.split(image)
    channel_delta = (
        np.abs(r_channel.astype(np.int16) - g_channel.astype(np.int16)).mean()
        + np.abs(g_channel.astype(np.int16) - b_channel.astype(np.int16)).mean()
        + np.abs(r_channel.astype(np.int16) - b_channel.astype(np.int16)).mean()
    ) / 3.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    mean_saturation = float(saturation.mean())
    sat_p95 = float(np.percentile(saturation, 95))
    highlight_ratio = float((value > 220).mean())

    if channel_delta < 10.0 and mean_saturation < 18.0 and sat_p95 < 40.0:
        return "bw_manga"
    if mean_saturation > 80.0 and highlight_ratio > 0.08:
        return "three_d_comic"
    if mean_saturation > 40.0:
        return "color_comic"
    return "bw_manga" if channel_delta < 16.0 else "color_comic"
