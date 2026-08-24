"""A vectorised, tested implementation of the Kohonen Self-Organising Map."""

from __future__ import annotations

from kohonen.config import TrainingConfig
from kohonen.decay import DecaySchedule, ExponentialDecay
from kohonen.som import SOM, IterationCallback

__version__ = "0.1.0"

__all__ = [
    "SOM",
    "DecaySchedule",
    "ExponentialDecay",
    "IterationCallback",
    "TrainingConfig",
    "__version__",
]
