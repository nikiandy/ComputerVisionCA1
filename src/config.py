"""Configuration values for the O-ring inspection pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime-tunable thresholds and processing parameters."""

    morphology_kernel_size: int = 3
    morphology_iterations: int = 1
    opening_kernel_size: int = 0
    angle_step_degrees: int = 2
    max_gap_fraction: float = 0.015
    max_thickness_std: float = 1.40
    max_outlier_fraction: float = 0.12
    outlier_sigma: float = 2.5
    min_valid_rays: int = 80
    min_foreground_ratio: float = 0.02
    max_foreground_ratio: float = 0.75
