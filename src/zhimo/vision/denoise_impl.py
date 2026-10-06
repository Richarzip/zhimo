"""Deterministic, stroke-aware restoration of uploaded calligraphy images.

Diagnostics are heuristics, not probabilities or authorship accuracy estimates.
All processing is local and preserves the oriented dimensions and ink polarity.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from .preprocess_impl import detect_and_fix_inversion


def _paper_field(gray: np.ndarray) -> np.ndarray:
    """Estimate paper brightness using upper tile percentiles on a small preview.

    Dense artwork and coloured paper can violate the sparse-ink assumption;
    callers therefore bound the strength of background correction.
    """
    h, w = gray.shape
    scale = min(1.0, 768 / max(h, w))
    preview = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA) if scale < 1 else gray
    ph, pw = preview.shape
    rows, cols = min(12, max(1, ph // 32)), min(12, max(1, pw // 32))
    ys = np.linspace(0, ph, rows + 1, dtype=int)
    xs = np.linspace(0, pw, cols + 1, dtype=int)
    tiles = np.empty((rows, cols), np.float32)
    for row in range(rows):
        for col in range(cols):
            tiles[row, col] = np.percentile(preview[ys[row]:ys[row + 1], xs[col]:xs[col + 1]], 85)
    smooth = cv2.GaussianBlur(tiles, (3, 3), 0.6, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(smooth, (w, h), interpolation=cv2.INTER_LINEAR)


def _diagnose(rgb: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    median = cv2.medianBlur(gray, 3)
    # A line, even one pixel wide, has a similar neighbour. Only isolated
    # outliers qualify; ambiguous dots and connected speckles are left alone.
    neighbours = np.ones((3, 3), np.uint8)
    neighbours[1, 1] = 0
    low = cv2.erode(gray, neighbours).astype(np.int16)
    high = cv2.dilate(gray, neighbours).astype(np.int16)
    signed = gray.astype(np.int16)
    impulse = ((low - signed) > 45) | ((signed - high) > 45)
    stable = gray.copy()
    stable[impulse] = median[impulse]
    field = _paper_field(stable)
    # MAD of a 2x2 high-pass response (L2 norm 2), in locally flat regions,
    # excluding isolated outliers. Estimate noise at the original pixel scale.
    local_range = cv2.morphologyEx(median, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    flat = local_range <= max(8.0, float(np.percentile(local_range, 60)))
    flat &= cv2.dilate(impulse.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
    response = cv2.filter2D(gray, cv2.CV_32F, np.array([[1, -1], [-1, 1]], np.float32))
    sigma = float(np.median(np.abs(response[flat])) / 1.34898) if np.count_nonzero(flat) >= 32 else 0.0
    darkness = field - stable.astype(np.float32)
    ink = (darkness > max(12.0, 4 * sigma)) & ~impulse
    contrast = float(np.median(darkness[ink])) if np.any(ink) else 0.0
    metrics = {
        "noise_sigma": round(sigma, 3),
        "impulse_ratio": round(float(np.mean(impulse)), 6),
        "background_variation": round(float(np.percentile(field, 90) - np.percentile(field, 10)), 3),
        "ink_contrast": round(contrast, 3),
        "ink_fraction": round(float(np.mean(ink)), 6),
        "laplacian_variance": round(float(cv2.Laplacian(gray, cv2.CV_32F).var()), 3),
    }
    return metrics, impulse, ink, field


def _blend(original: np.ndarray, candidate: np.ndarray, amount: np.ndarray | float) -> np.ndarray:
    # Avoid several full RGB float buffers on large scans.
    output = np.empty_like(original)
    for channel in range(3):
        source = original[..., channel].astype(np.float32)
        value = source + amount * (candidate[..., channel].astype(np.float32) - source)
        output[..., channel] = np.clip(np.rint(value), 0, 255).astype(np.uint8)
    return output


def _basic(rgb: np.ndarray) -> np.ndarray:
    """The previous fixed recipe, retained for reproducible comparisons."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    median = cv2.medianBlur(bgr, 3)
    bilateral = cv2.bilateralFilter(median, d=5, sigmaColor=35, sigmaSpace=35)
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def denoise_image(image: Image.Image, *, mode: str = "adaptive") -> tuple[Image.Image, dict[str, Any]]:
    """Return an RGB image and JSON-compatible diagnostics, without training.

    Adaptive mode skips clean/ambiguous images and protects detected strokes.
    Basic mode reproduces the former median/bilateral/CLAHE transformation.
    Apply EXIF orientation once. Temporarily normalize reversed rubbings for
    diagnosis and adaptive processing, restoring polarity for later detection.
    """
    if mode not in {"adaptive", "basic"}:
        raise ValueError("denoise mode must be 'adaptive' or 'basic'")
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    rgb = np.asarray(oriented)
    normalized, inversion = detect_and_fix_inversion(oriented)
    source = np.asarray(normalized)
    before, impulse, ink, field = _diagnose(source)
    info: dict[str, Any] = {
        "enabled": True, "mode": mode, "applied": False, "method": "identity",
        "reason": "no_actionable_degradation", "steps": [], "parameters": {},
        "quality_before": before, "quality_after": before.copy(),
        "stroke_change": 0.0, "stroke_retention": 1.0 if np.any(ink) else None,
        "polarity_normalized": inversion["was_inverted"],
    }
    if mode == "basic":
        output = _basic(rgb)
        measured = 255 - output if inversion["was_inverted"] else output
        info.update(applied=bool(np.any(output != rgb)), method="median3 + bilateral + CLAHE",
                    reason="basic_requested", steps=["median3", "bilateral", "CLAHE"],
                    parameters={"median_kernel": 3, "bilateral_d": 5, "sigma_color": 35,
                                "sigma_space": 35, "clahe_clip": 1.5, "clahe_grid": [8, 8]})
    elif min(source.shape[:2]) < 16:
        info["reason"] = "image_too_small"
        return Image.fromarray(rgb.copy()), info
    else:
        measured = source.copy()
        steps, parameters = info["steps"], info["parameters"]
        if before["impulse_ratio"] >= 0.0005 and np.count_nonzero(impulse) >= 8:
            replacement = cv2.medianBlur(measured, 3)
            measured[impulse] = replacement[impulse]
            steps.append("selective_median")
            parameters["impulse_kernel"] = 3

        # Feathered protection includes the ink and its immediately adjacent edge.
        protection = cv2.GaussianBlur(cv2.dilate(ink.astype(np.float32), np.ones((3, 3), np.uint8)),
                                      (5, 5), 0.8)
        protection[ink] = 1.0
        sigma = before["noise_sigma"]
        if sigma >= 3.0:
            diameter = 5 if min(source.shape[:2]) < 1000 else 7
            sigma_color = float(np.clip(3 * sigma, 12, 45))
            filtered = cv2.bilateralFilter(measured, diameter, sigma_color, diameter / 2)
            measured = _blend(measured, filtered, 1 - 0.9 * protection)
            steps.append("protected_bilateral")
            parameters.update(bilateral_d=diameter, sigma_color=round(sigma_color, 3),
                              sigma_space=diameter / 2, ink_protection=0.9)

        if before["background_variation"] > 18 and 0.002 < before["ink_fraction"] < 0.4:
            target = float(np.percentile(field, 85))
            gain = np.clip(target / np.maximum(field, 32), 1, 1.25)
            # Use one smooth gain across both sides of an ink edge. Blending
            # illumination correction with an ink mask would create dark halos.
            # Dark ink changes less in absolute value under multiplicative gain.
            for channel in range(3):
                measured[..., channel] = np.clip(np.rint(measured[..., channel] * gain), 0, 255).astype(np.uint8)
            steps.append("background_normalization")
            parameters["max_background_gain"] = 1.25

        if 15 < before["ink_contrast"] < 75 and 0.002 < before["ink_fraction"] < 0.4:
            lab = cv2.cvtColor(measured, cv2.COLOR_RGB2LAB)
            # A bounded luminance stretch darkens faint ink without globally
            # equalizing (and amplifying) the paper texture. Leave a/b intact.
            gray = cv2.cvtColor(measured, cv2.COLOR_RGB2GRAY)
            delta = np.clip((field - gray) * 0.15, 0, 8) * protection
            lab[..., 0] = np.clip(np.rint(lab[..., 0] - delta), 0, 255).astype(np.uint8)
            measured = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            steps.append("mild_contrast")
            parameters.update(contrast_strength=0.15, max_lightness_change=8)

        if not steps:
            return Image.fromarray(rgb.copy()), info
        info.update(reason="degradation_detected", method=" + ".join(steps))

    after, _, _, after_field = _diagnose(measured)
    if np.any(ink):
        before_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
        after_gray = cv2.cvtColor(measured, cv2.COLOR_RGB2GRAY).astype(np.float32)
        change = float(np.mean(np.abs(after_gray[ink] - before_gray[ink])) / 255)
        # Re-use original ink locations; re-thresholding alone could hide loss.
        retention = float(np.mean((after_field - after_gray)[ink] >= 0.7 * (field - before_gray)[ink]))
        info.update(stroke_change=round(change, 6), stroke_retention=round(retention, 6))
        if mode == "adaptive" and (change > 0.06 or retention < 0.97):
            info["rejected_candidate"] = {
                "steps": info["steps"].copy(), "quality": after,
                "stroke_change": round(change, 6), "stroke_retention": round(retention, 6),
            }
            info.update(reason="stroke_guard_reverted", method="identity", steps=[],
                        stroke_change=0.0, stroke_retention=1.0)
            return Image.fromarray(rgb.copy()), info
    info["quality_after"] = after
    if mode == "adaptive":
        output = 255 - measured if inversion["was_inverted"] else measured
        info["applied"] = bool(np.any(output != rgb))
    return Image.fromarray(output), info
