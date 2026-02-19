"""I/O, annotation and visualisation helpers using allowed cv2 calls only."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_bgr_image(path: Path) -> Optional[np.ndarray]:
    """Load color image as BGR. Returns None if unreadable."""
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def save_image(path: Path, image: np.ndarray) -> bool:
    """Save image to disk, creating parent directory as needed."""
    ensure_dir(path.parent)
    return bool(cv2.imwrite(str(path), image))


def binary_to_uint8(mask: np.ndarray) -> np.ndarray:
    """Convert bool/{0,1} mask to uint8 {0,255}."""
    return ((mask > 0).astype(np.uint8)) * 255


def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    """Convert grayscale to 3-channel BGR using numpy stacking."""
    g = gray.astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    """Convert binary mask to displayable BGR image."""
    return gray_to_bgr(binary_to_uint8(mask))


def label_map_to_color(labels: np.ndarray, focus_label: int | None = None) -> np.ndarray:
    """Create deterministic pseudo-color visualisation for label map."""
    h, w = labels.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    fg = labels > 0
    if np.any(fg):
        v = labels[fg].astype(np.uint32)
        colored[fg] = np.stack(
            [
                (v * 37) % 256,
                (v * 83) % 256,
                (v * 157) % 256,
            ],
            axis=1,
        ).astype(np.uint8)
    if focus_label is not None:
        colored[labels == focus_label] = np.array([0, 255, 255], dtype=np.uint8)
    return colored


def annotate_image(
    image_bgr: np.ndarray,
    lines: Iterable[str],
    passed: bool,
    bbox: Tuple[int, int, int, int] | None = None,
    centroid_xy: Tuple[float, float] | None = None,
) -> np.ndarray:
    """Overlay result text and simple shapes onto the image."""
    out = image_bgr.copy()
    lines_list = list(lines)
    if not lines_list:
        return out

    text_bg_top_left = (8, 8)
    text_bg_bottom_right = (610, 8 + 24 * len(lines_list))
    cv2.rectangle(out, text_bg_top_left, text_bg_bottom_right, (0, 0, 0), -1)

    status_color = (0, 200, 0) if passed else (0, 0, 255)
    for idx, text in enumerate(lines_list):
        y = 28 + idx * 22
        color = status_color if idx == 0 else (255, 255, 255)
        cv2.putText(
            out,
            text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    if bbox is not None:
        min_x, min_y, max_x, max_y = bbox
        cv2.rectangle(out, (min_x, min_y), (max_x, max_y), status_color, 2)

    if centroid_xy is not None:
        cx = int(round(centroid_xy[0]))
        cy = int(round(centroid_xy[1]))
        cv2.circle(out, (cx, cy), 4, (255, 255, 0), -1)

    return out
