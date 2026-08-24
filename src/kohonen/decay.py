"""Decay schedules for the neighbourhood radius and the learning rate."""

from __future__ import annotations

import math
import warnings
from typing import Protocol, runtime_checkable

MIN_RADIUS: float = 1.0 + 1e-9
"""Floor for the initial radius.

The brief derives ``lambda = n_iterations / log(sigma_0)``, which is undefined
at ``sigma_0 == 1`` (a 2x2 grid) and negative below it (a 1x1 grid) -- where a
negative time constant makes the neighbourhood *grow* instead of decay. Small
grids are legitimate input, so the radius is clamped just above 1 rather than
rejected.
"""


@runtime_checkable
class DecaySchedule(Protocol):
    """Maps an iteration number to a decayed value."""

    def __call__(self, t: int) -> float: ...


def time_constant(n_iterations: int, initial_radius: float) -> float:
    """Return ``lambda``, the shared time constant from the brief.

    Warns and clamps when ``initial_radius <= 1``, which would otherwise make
    the time constant undefined or negative.
    """
    if n_iterations <= 0:
        raise ValueError(f"n_iterations must be positive, got {n_iterations}")
    radius = initial_radius
    if radius <= MIN_RADIUS:
        warnings.warn(
            f"initial radius {initial_radius} is too small for a stable time "
            f"constant (grid is 2x2 or smaller); clamping to {MIN_RADIUS}",
            UserWarning,
            stacklevel=2,
        )
        radius = MIN_RADIUS
    return n_iterations / math.log(radius)


def clamped_radius(initial_radius: float) -> float:
    """Return the initial radius, floored at ``MIN_RADIUS``.

    Callers must use this to seed the radius decay, so that a grid small
    enough to trigger the clamp in :func:`time_constant` also decays from the
    clamped value. Clamping one but not the other would leave the two
    inconsistent.
    """
    return max(initial_radius, MIN_RADIUS)


class ExponentialDecay:
    """``value(t) = initial_value * exp(-t / time_constant)``."""

    __slots__ = ("initial_value", "time_constant")

    def __init__(self, initial_value: float, time_constant: float) -> None:
        self.initial_value = initial_value
        self.time_constant = time_constant

    def __call__(self, t: int) -> float:
        return self.initial_value * math.exp(-t / self.time_constant)
