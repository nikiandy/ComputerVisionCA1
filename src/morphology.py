from __future__ import annotations

import numpy as np


def _validate_kernel_size(kernel_size: int) -> None:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")


def _as_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def _shift_with_zero_padding(binary: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift image by (dy, dx) with zeros filled outside image bounds"""
    h, w = binary.shape
    shifted = np.zeros_like(binary, dtype=np.uint8)

    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)

    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)

    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = binary[src_y0:src_y1, src_x0:src_x1]
    return shifted


#  grows the white regions
def binary_dilation(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Binary dilation with a square structuring element"""
    _validate_kernel_size(kernel_size)
    binary = _as_binary(mask)
    radius = kernel_size // 2
    out = np.zeros_like(binary, dtype=np.uint8)

    # any hit makes 1
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = _shift_with_zero_padding(binary, dy, dx)
            out = np.maximum(out, shifted)

    return out


# shrinks the white regions
def binary_erosion(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Binary erosion with a square structuring element"""
    _validate_kernel_size(kernel_size)
    binary = _as_binary(mask)
    radius = kernel_size // 2
    out = np.ones_like(binary, dtype=np.uint8)

    # all hits must be 1
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = _shift_with_zero_padding(binary, dy, dx)
            out = np.minimum(out, shifted)

    return out


# fills small holes in the white regions
def binary_closing(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Binary closing: dilation followed by erosion"""
    out = _as_binary(mask)
    for _ in range(max(0, iterations)):
        # close tiny holes
        out = binary_dilation(out, kernel_size=kernel_size)
        out = binary_erosion(out, kernel_size=kernel_size)
    return out


# removes small objects from the white regions
def binary_opening(mask: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Binary opening: erosion followed by dilation"""
    out = _as_binary(mask)
    for _ in range(max(0, iterations)):
        out = binary_erosion(out, kernel_size=kernel_size)
        out = binary_dilation(out, kernel_size=kernel_size)
    return out
