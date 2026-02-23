"""Histogram-based thresholding routines implemented with numpy only."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def bgr_to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to uint8 grayscale without cv2.cvtColor."""
    # split channels
    b = image_bgr[:, :, 0].astype(np.float32)
    g = image_bgr[:, :, 1].astype(np.float32)
    r = image_bgr[:, :, 2].astype(np.float32)
    # weighted gray value
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return np.clip(gray, 0, 255).astype(np.uint8)


def histogram_256(gray: np.ndarray) -> np.ndarray:
    """Return a 256-bin histogram of grayscale values."""
    return np.bincount(gray.ravel(), minlength=256).astype(np.int64)


def otsu_threshold_from_histogram(hist: np.ndarray) -> int:
    """Compute Otsu threshold using only histogram counts."""
    total = int(hist.sum())
    if total == 0:
        return 0

    intensity = np.arange(256, dtype=np.float64)
    sum_total = float(np.dot(intensity, hist))
    sum_background = 0.0
    weight_background = 0.0
    best_threshold = 0
    best_between_variance = -1.0

    for threshold in range(256):
        count = float(hist[threshold])
        weight_background += count
        if weight_background == 0.0:
            continue

        weight_foreground = float(total) - weight_background
        if weight_foreground == 0.0:
            break

        # means for this split
        sum_background += threshold * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        mean_delta = mean_background - mean_foreground
        between_variance = (
            weight_background * weight_foreground * mean_delta * mean_delta
        )

        # keep best threshold
        if between_variance > best_between_variance:
            best_between_variance = between_variance
            best_threshold = threshold

    return int(best_threshold)


def otsu_threshold(gray: np.ndarray) -> Tuple[int, np.ndarray]:
    """Return (threshold, histogram) for a grayscale image."""
    hist = histogram_256(gray)
    threshold = otsu_threshold_from_histogram(hist)
    return threshold, hist


def threshold_with_auto_polarity(
    gray: np.ndarray,
    threshold: int,
    min_foreground_ratio: float = 0.02,
    max_foreground_ratio: float = 0.75,
) -> Tuple[np.ndarray, bool, float]:
    """
    Apply threshold and auto-fix foreground polarity.

    Returns:
        binary_mask (uint8 in {0,1}), inverted_flag, foreground_ratio
    """
    dark_foreground = gray <= threshold
    light_foreground = gray >= threshold

    dark_ratio = float(dark_foreground.mean())
    light_ratio = float(light_foreground.mean())

    dark_fg_mean = float(gray[dark_foreground].mean()) if dark_foreground.any() else 255.0
    dark_bg_mean = (
        float(gray[~dark_foreground].mean()) if (~dark_foreground).any() else 0.0
    )

    # assume ring is dark
    use_light_foreground = False

    # flip if dark mask looks wrong
    if dark_fg_mean >= dark_bg_mean:
        use_light_foreground = True
    elif not (min_foreground_ratio <= dark_ratio <= max_foreground_ratio):
        # pick ratio closer to normal
        target_ratio = 0.25
        if abs(light_ratio - target_ratio) < abs(dark_ratio - target_ratio):
            use_light_foreground = True

    if use_light_foreground:
        mask = light_foreground.astype(np.uint8)
        return mask, True, light_ratio

    mask = dark_foreground.astype(np.uint8)
    return mask, False, dark_ratio
