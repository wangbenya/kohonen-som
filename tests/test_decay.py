from __future__ import annotations

import math
from itertools import pairwise

import pytest

from kohonen.decay import (
    MIN_RADIUS,
    ExponentialDecay,
    clamped_radius,
    time_constant,
)


def test_time_constant_matches_the_brief() -> None:
    assert time_constant(100, 5.0) == pytest.approx(100 / math.log(5.0))


@pytest.mark.parametrize("initial_radius", [1.0, 0.5, 0.25])
def test_degenerate_radius_is_clamped_not_raised(initial_radius: float) -> None:
    """A 1x1 or 2x2 grid is legitimate input, not a user error."""
    with pytest.warns(UserWarning, match="initial radius"):
        tau = time_constant(100, initial_radius)
    assert tau > 0
    assert math.isfinite(tau)


def test_clamp_keeps_decay_monotonically_decreasing() -> None:
    with pytest.warns(UserWarning):
        tau = time_constant(50, 1.0)
    decay = ExponentialDecay(MIN_RADIUS, tau)
    values = [decay(t) for t in range(50)]
    assert all(b <= a for a, b in pairwise(values))


def test_clamped_radius_floors_degenerate_values() -> None:
    assert clamped_radius(0.5) == MIN_RADIUS
    assert clamped_radius(1.0) == MIN_RADIUS


def test_clamped_radius_leaves_normal_values_untouched() -> None:
    assert clamped_radius(5.0) == 5.0


def test_exponential_decay_starts_at_initial_value() -> None:
    assert ExponentialDecay(5.0, 62.0)(0) == pytest.approx(5.0)


def test_exponential_decay_matches_closed_form() -> None:
    decay = ExponentialDecay(0.1, 62.0)
    assert decay(10) == pytest.approx(0.1 * math.exp(-10 / 62.0))


def test_exponential_decay_is_strictly_decreasing() -> None:
    decay = ExponentialDecay(5.0, 62.0)
    values = [decay(t) for t in range(100)]
    assert all(b < a for a, b in pairwise(values))


def test_time_constant_rejects_non_positive_iterations() -> None:
    with pytest.raises(ValueError, match="n_iterations"):
        time_constant(0, 5.0)
