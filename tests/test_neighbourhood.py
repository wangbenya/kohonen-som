from __future__ import annotations

import numpy as np
import pytest

from kohonen.neighbourhood import (
    coordinate_grid,
    gaussian_influence,
    squared_distance_to,
)


def test_coordinate_grid_uses_matrix_indexing() -> None:
    gx, gy = coordinate_grid(3, 2)
    np.testing.assert_array_equal(gx, [[0, 0], [1, 1], [2, 2]])
    np.testing.assert_array_equal(gy, [[0, 1], [0, 1], [0, 1]])


def test_squared_distance_is_zero_at_the_bmu() -> None:
    gx, gy = coordinate_grid(4, 4)
    d2 = squared_distance_to(gx, gy, 2, 1)
    assert d2[2, 1] == 0


def test_squared_distance_matches_pythagoras() -> None:
    gx, gy = coordinate_grid(4, 4)
    d2 = squared_distance_to(gx, gy, 0, 0)
    assert d2[3, 3] == 18  # 3^2 + 3^2


def test_influence_is_one_at_the_bmu() -> None:
    influence = gaussian_influence(np.zeros((3, 3)), sigma=2.0)
    np.testing.assert_allclose(influence, np.ones((3, 3)))


def test_influence_decreases_with_distance() -> None:
    gx, gy = coordinate_grid(5, 5)
    d2 = squared_distance_to(gx, gy, 0, 0)
    influence = gaussian_influence(d2, sigma=2.0)
    assert influence[0, 0] > influence[0, 1] > influence[0, 4]


def test_influence_matches_closed_form() -> None:
    d2 = np.array([[4.0]])
    result = gaussian_influence(d2, sigma=2.0)
    assert result[0, 0] == pytest.approx(np.exp(-4.0 / (2 * 4.0)))


def test_influence_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma"):
        gaussian_influence(np.zeros((2, 2)), sigma=0.0)
