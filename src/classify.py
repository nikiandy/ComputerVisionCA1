from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.config import PipelineConfig


@dataclass
class ClassificationResult:
    passed: bool
    gap_fraction: float
    thickness_mean: float
    thickness_std: float
    outlier_fraction: float
    valid_rays: int
    total_rays: int
    reason: str


def _ray_first_last_hit(
    binary_mask: np.ndarray,
    centroid_xy: Tuple[float, float],
    angle_deg: float,
) -> Tuple[int | None, int | None]:
    """Sample one ray and return first and last foreground hit distances"""
    h, w = binary_mask.shape
    cx, cy = centroid_xy
    theta = np.deg2rad(angle_deg)
    dx = float(np.cos(theta))
    dy = float(np.sin(theta))
    # long enough to hit image edge
    max_steps = int(np.hypot(h, w)) + 2

    first_hit = None
    last_hit = None
    prev_xy = None

    for step in range(max_steps):
        x = int(round(cx + step * dx))
        y = int(round(cy + step * dy))
        if x < 0 or x >= w or y < 0 or y >= h:
            break

        # skip same rounded pixel
        xy = (x, y)
        if xy == prev_xy:
            continue
        prev_xy = xy

        if binary_mask[y, x] > 0:
            if first_hit is None:
                first_hit = step
            last_hit = step

    return first_hit, last_hit


def radial_thickness_profile(
    ring_mask: np.ndarray,
    centroid_xy: Tuple[float, float],
    angle_step_degrees: int = 2,
) -> np.ndarray:
    """Compute radial thickness for multiple rays around the centroid"""
    if angle_step_degrees <= 0:
        raise ValueError("angle_step_degrees must be > 0")

    angles = np.arange(0, 360, angle_step_degrees, dtype=np.float32)
    thickness_values = np.full(angles.shape[0], np.nan, dtype=np.float32)

    for idx, angle in enumerate(angles):
        first_hit, last_hit = _ray_first_last_hit(ring_mask, centroid_xy, float(angle))
        if first_hit is None or last_hit is None:
            continue
        # ring width on this ray
        thickness_values[idx] = float(last_hit - first_hit)

    return thickness_values


# classifies the ring as pass/fail using radial thickness consistency
def classify_oring(
    ring_mask: np.ndarray,
    centroid_xy: Tuple[float, float],
    config: PipelineConfig,
) -> ClassificationResult:
    thickness = radial_thickness_profile(
        ring_mask=ring_mask,
        centroid_xy=centroid_xy,
        angle_step_degrees=config.angle_step_degrees,
    )

    total_rays = int(thickness.size)
    valid = thickness[~np.isnan(thickness)]
    valid_rays = int(valid.size)
    # no hit means gap
    gap_fraction = 1.0 - (valid_rays / max(1, total_rays))

    if valid_rays == 0:
        return ClassificationResult(
            passed=False,
            gap_fraction=1.0,
            thickness_mean=0.0,
            thickness_std=0.0,
            outlier_fraction=1.0,
            valid_rays=0,
            total_rays=total_rays,
            reason="no foreground hits along rays",
        )

    thickness_mean = float(np.mean(valid))
    thickness_std = float(np.std(valid))
    # simple outlier rule
    outlier_threshold = max(config.outlier_sigma * thickness_std, 2.0)
    outlier_fraction = float(np.mean(np.abs(valid - thickness_mean) > outlier_threshold))

    # fail reasons list is used to store the reasons for the ring to be failed
    fail_reasons = []
    if gap_fraction > config.max_gap_fraction:
        fail_reasons.append("gap_fraction above threshold")
    if valid_rays < config.min_valid_rays:
        fail_reasons.append("not enough valid rays")
    if thickness_std > config.max_thickness_std:
        fail_reasons.append("thickness std too high")
    if outlier_fraction > config.max_outlier_fraction:
        fail_reasons.append("too many thickness outliers")

    passed = len(fail_reasons) == 0
    reason = "ok" if passed else "; ".join(fail_reasons)
    return ClassificationResult(
        passed=passed,
        gap_fraction=gap_fraction,
        thickness_mean=thickness_mean,
        thickness_std=thickness_std,
        outlier_fraction=outlier_fraction,
        valid_rays=valid_rays,
        total_rays=total_rays,
        reason=reason,
    )
