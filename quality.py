"""Simple, explainable image-quality measurements."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def assess_quality(image: np.ndarray) -> dict[str, Any]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    noise_estimate = float(cv2.Laplacian(gray, cv2.CV_64F).std())
    flags: list[str] = []

    if width < 800 or height < 600:
        flags.append("Low resolution may limit feature matching.")
    if blur_score < 80:
        flags.append("Image appears blurred, copy-move matches may be unreliable.")
    if brightness < 40:
        flags.append("Image appears underexposed.")
    elif brightness > 220:
        flags.append("Image appears overexposed.")

    return {
        "width": width,
        "height": height,
        "blur_laplacian_variance": round(blur_score, 2),
        "mean_brightness": round(brightness, 2),
        "noise_proxy": round(noise_estimate, 2),
        "flags": flags,
    }
