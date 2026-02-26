from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ComponentStats:
    label: int
    area: int
    bbox: Tuple[int, int, int, int]  # min_x min_y max_x max_y
    centroid: Tuple[float, float]  # x y


def connected_components(binary_mask: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, List[ComponentStats]]:
    """
    Label connected components in a binary image.

    Args:
        binary_mask: uint8/bool array, non-zero means foreground.
        connectivity: 4 or 8 neighborhood.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    binary = (binary_mask > 0)
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    components: List[ComponentStats] = []

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    current_label = 0
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or labels[y, x] != 0:
                continue

            # new component
            current_label += 1
            labels[y, x] = current_label
            stack = [(y, x)]

            area = 0
            min_x = x
            min_y = y
            max_x = x
            max_y = y
            sum_x = 0.0
            sum_y = 0.0

            while stack:
                cy, cx = stack.pop()
                area += 1
                sum_x += cx
                sum_y += cy

                if cx < min_x:
                    min_x = cx
                if cy < min_y:
                    min_y = cy
                if cx > max_x:
                    max_x = cx
                if cy > max_y:
                    max_y = cy

                for dy, dx in neighbors:
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if not binary[ny, nx] or labels[ny, nx] != 0:
                        continue
                    # grow region
                    labels[ny, nx] = current_label
                    stack.append((ny, nx))

            # save stats
            centroid = (sum_x / area, sum_y / area)
            components.append(
                ComponentStats(
                    label=current_label,
                    area=area,
                    bbox=(min_x, min_y, max_x, max_y),
                    centroid=centroid,
                )
            )

    return labels, components


def largest_component(components: List[ComponentStats]) -> Optional[ComponentStats]:
    """Return the component with maximum area or None if empty"""
    if not components:
        return None
    return max(components, key=lambda c: c.area)
