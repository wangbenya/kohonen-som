"""Grid geometry and the Gaussian neighbourhood function.

The coordinate grid is built once per training run and reused for every
sample. Rebuilding it inside the sample loop -- or worse, walking it with
nested Python loops -- is what made the original implementation slow.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def coordinate_grid(
    width: int, height: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return per-node x and y coordinates, each shaped ``(width, height)``.

    Uses ``indexing="ij"`` so axis 0 is x and axis 1 is y, matching the
    ``weights[x, y]`` convention of the original implementation.
    """
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.int64),
        np.arange(height, dtype=np.int64),
        indexing="ij",
    )
    return grid_x, grid_y


def squared_distance_to(
    grid_x: NDArray[np.int64],
    grid_y: NDArray[np.int64],
    bmu_x: int,
    bmu_y: int,
) -> NDArray[np.int64]:
    """Return the squared grid distance from every node to the BMU.

    Squared distance is returned deliberately: the Gaussian immediately
    squares it again, so taking a square root here would be wasted work.
    """
    dx = grid_x - bmu_x
    dy = grid_y - bmu_y
    return dx * dx + dy * dy


def gaussian_influence(
    squared_distance: NDArray[np.floating] | NDArray[np.int64],
    sigma: float,
) -> NDArray[np.float64]:
    """Return ``exp(-d^2 / (2 * sigma^2))`` for every node."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    result: NDArray[np.float64] = np.exp(-squared_distance / (2.0 * sigma * sigma))
    return result
