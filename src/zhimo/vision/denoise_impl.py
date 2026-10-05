"""Traditional computer-vision denoising for uploaded calligraphy images.

The recognizer still receives a PIL image.  This module deliberately avoids
learned restoration models so the preprocessing step is deterministic and
works offline with OpenCV.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


def denoise_image(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    """Denoise a scanned/photo image while preserving ink edges.

    A median pass removes isolated speckles, bilateral filtering suppresses
    low-level noise without flattening strokes, and a conservative CLAHE pass
    restores local contrast after filtering. The parameters are intentionally
    fixed so model comparisons remain reproducible.
    """
    rgb = np.asarray(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    median = cv2.medianBlur(bgr, 3)
    bilateral = cv2.bilateralFilter(median, d=5, sigmaColor=35, sigmaSpace=35)
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = cv2.merge((clahe.apply(lightness), a_channel, b_channel))
    output = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    return Image.fromarray(output), {
        "enabled": True,
        "method": "median3 + bilateral + CLAHE",
        "preserves_edges": True,
    }

