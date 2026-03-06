from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.cc_label import connected_components, largest_component
from src.classify import classify_oring
from src.config import PipelineConfig
from src.io_utils import (
    annotate_image,
    ensure_dir,
    gray_to_bgr,
    label_map_to_color,
    load_bgr_image,
    mask_to_bgr,
    save_image,
)
from src.morphology import binary_closing, binary_opening
from src.thresholding import bgr_to_grayscale, otsu_threshold, threshold_with_auto_polarity


PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageSummary:
    filename: str
    threshold: Optional[int]
    result: str
    time_ms: float
    gap_fraction: Optional[float]
    thickness_std: Optional[float]
    outlier_fraction: Optional[float]
    note: str


def resolve_path(path_arg: str) -> Path:
    """Resolve absolute/relative path arguments robustly"""
    p = Path(path_arg)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def list_image_files(images_path: Path) -> list[Path]:
    """Get sorted input image paths"""
    if not images_path.exists():
        return []
    if images_path.is_file():
        return [images_path] if images_path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    return sorted(
        [
            p
            for p in images_path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


def process_image(
    image_path: Path,
    output_dir: Path,
    debug_dir: Optional[Path],
    config: PipelineConfig,
) -> ImageSummary:
    """Run full O-ring pipeline for one image"""
    # start the timer
    t0 = time.perf_counter()
    # load image
    image = load_bgr_image(image_path)
    if image is None:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return ImageSummary(
            filename=image_path.name,
            threshold=None,
            result="SKIP",
            time_ms=elapsed,
            gap_fraction=None,
            thickness_std=None,
            outlier_fraction=None,
            note="unreadable image",
        )

    # grayscale img
    gray = bgr_to_grayscale(image)
    threshold, _hist = otsu_threshold(gray)
    raw_mask, inverted, fg_ratio = threshold_with_auto_polarity(
        gray=gray,
        threshold=threshold,
        min_foreground_ratio=config.min_foreground_ratio,
        max_foreground_ratio=config.max_foreground_ratio,
    )

    cleaned = binary_closing(
        raw_mask,
        kernel_size=config.morphology_kernel_size,
        iterations=config.morphology_iterations,
    )
    if config.opening_kernel_size > 0:
        cleaned = binary_opening(cleaned, kernel_size=config.opening_kernel_size, iterations=1)

    # labels connected components
    labels, components = connected_components(cleaned, connectivity=8)

    # find the largest component
    largest = largest_component(components)

    if largest is None:
        elapsed = (time.perf_counter() - t0) * 1000.0
        lines = [
            "FAIL",
            f"thr={threshold} inv={int(inverted)} time={elapsed:.2f} ms",
            f"fg_ratio={fg_ratio:.3f} (no component found)",
        ]
        annotated = annotate_image(image, lines, passed=False)
        save_image(output_dir / image_path.name, annotated)
        if debug_dir is not None:
            save_image(debug_dir / f"{image_path.stem}_01_gray.png", gray_to_bgr(gray))
            save_image(debug_dir / f"{image_path.stem}_02_binary_raw.png", mask_to_bgr(raw_mask))
            save_image(debug_dir / f"{image_path.stem}_03_binary_clean.png", mask_to_bgr(cleaned))
            save_image(debug_dir / f"{image_path.stem}_04_labels.png", label_map_to_color(labels))
        return ImageSummary(
            filename=image_path.name,
            threshold=threshold,
            result="FAIL",
            time_ms=elapsed,
            gap_fraction=1.0,
            thickness_std=0.0,
            outlier_fraction=1.0,
            note="no component found",
        )

    ring_mask_clean = (labels == largest.label).astype(np.uint8)
    # keep only raw ring pixels
    analysis_mask = ((raw_mask > 0) & (ring_mask_clean > 0)).astype(np.uint8)
    if int(np.sum(analysis_mask)) == 0:
        analysis_mask = ring_mask_clean

    classification = classify_oring(
        ring_mask=analysis_mask,
        centroid_xy=largest.centroid,
        config=config,
    )
    # Calculate elapsed time
    elapsed = (time.perf_counter() - t0) * 1000.0

    status = "PASS" if classification.passed else "FAIL"
    lines = [
        status,
        f"thr={threshold} inv={int(inverted)} time={elapsed:.2f} ms",
        (
            "gap={:.3f} std={:.2f} out={:.3f}".format(
                classification.gap_fraction,
                classification.thickness_std,
                classification.outlier_fraction,
            )
        ),
    ]
    annotated = annotate_image(
        image_bgr=image,
        lines=lines,
        passed=classification.passed,
        bbox=largest.bbox,
        centroid_xy=largest.centroid,
    )
    save_image(output_dir / image_path.name, annotated)

    if debug_dir is not None:
        save_image(debug_dir / f"{image_path.stem}_01_gray.png", gray_to_bgr(gray))
        save_image(debug_dir / f"{image_path.stem}_02_binary_raw.png", mask_to_bgr(raw_mask))
        save_image(debug_dir / f"{image_path.stem}_03_binary_clean.png", mask_to_bgr(cleaned))
        save_image(
            debug_dir / f"{image_path.stem}_04_labels.png",
            label_map_to_color(labels, focus_label=largest.label),
        )
        save_image(debug_dir / f"{image_path.stem}_05_ring_mask_clean.png", mask_to_bgr(ring_mask_clean))
        save_image(debug_dir / f"{image_path.stem}_06_ring_mask_analysis.png", mask_to_bgr(analysis_mask))

    return ImageSummary(
        filename=image_path.name,
        threshold=threshold,
        result=status,
        time_ms=elapsed,
        gap_fraction=classification.gap_fraction,
        thickness_std=classification.thickness_std,
        outlier_fraction=classification.outlier_fraction,
        note=classification.reason,
    )


def _fmt_float(value: Optional[float], precision: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{value:.{precision}f}"


def print_summary(rows: list[ImageSummary]) -> None:
    """Print neat per-image summary table"""
    header = (
        f"{'filename':<14} | {'threshold':>9} | {'result':<6} | {'time_ms':>9} | "
        f"{'gap_frac':>8} | {'thk_std':>8} | {'out_frac':>8} | note"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        threshold_str = "-" if row.threshold is None else str(row.threshold)
        print(
            f"{row.filename:<14} | "
            f"{threshold_str:>9} | "
            f"{row.result:<6} | "
            f"{row.time_ms:9.2f} | "
            f"{_fmt_float(row.gap_fraction, 3):>8} | "
            f"{_fmt_float(row.thickness_std, 2):>8} | "
            f"{_fmt_float(row.outlier_fraction, 3):>8} | "
            f"{row.note}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="O-ring inspection pipeline (numpy + limited cv2 I/O)")
    parser.add_argument(
        "--images",
        type=str,
        default="images",
        help="Input image folder or single image path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output folder for annotated results",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate pipeline images to output/debug",
    )
    return parser.parse_args()


def main() -> int:
    # read args
    args = parse_args()
    images_path = resolve_path(args.images)
    output_dir = resolve_path(args.output)
    debug_dir = output_dir / "debug" if args.debug else None

    # make output folders
    ensure_dir(output_dir)
    if debug_dir is not None:
        ensure_dir(debug_dir)

    input_images = list_image_files(images_path)
    if not input_images:
        print(f"No input images found at: {images_path}")
        return 1

    config = PipelineConfig()
    rows: list[ImageSummary] = []
    # run all images
    for image_path in input_images:
        row = process_image(
            image_path=image_path,
            output_dir=output_dir,
            debug_dir=debug_dir,
            config=config,
        )
        rows.append(row)

    print_summary(rows)
    print(f"\nAnnotated outputs saved to: {output_dir}")
    if debug_dir is not None:
        print(f"Debug outputs saved to: {debug_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
