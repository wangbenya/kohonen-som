"""A vectorised, tested implementation of the Kohonen Self-Organising Map."""

from __future__ import annotations

from kohonen.config import TrainingConfig
from kohonen.decay import DecaySchedule, ExponentialDecay

__version__ = "0.1.0"

__all__ = [
    "DecaySchedule",
    "ExponentialDecay",
    "TrainingConfig",
    "__version__",
]
