"""Training hyperparameters, validated at construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters for a SOM training run.

    Frozen and hashable so it can be logged wholesale as experiment
    parameters (for example, MLflow params) without defensive copying.

    Args:
        n_iterations: Number of passes over the training data. Zero is valid
            and yields an untrained map.
        initial_learning_rate: Learning rate at t=0.
        initial_radius: Neighbourhood radius at t=0. When ``None``, defaults
            to ``max(width, height) / 2`` as the brief specifies.
        seed: Seed for weight initialisation. ``None`` means non-reproducible.
        radius_cutoff: When set, restricts each update to a window of
            ``radius_cutoff * sigma_t`` nodes around the BMU. This is an
            approximation, so it is off by default; see ``docs/REVIEW.md``.
    """

    n_iterations: int = 100
    initial_learning_rate: float = 0.1
    initial_radius: float | None = None
    seed: int | None = None
    radius_cutoff: float | None = None

    def __post_init__(self) -> None:
        if self.n_iterations < 0:
            raise ValueError(
                f"n_iterations must be non-negative, got {self.n_iterations}"
            )
        if self.initial_learning_rate <= 0:
            raise ValueError(
                "initial_learning_rate must be positive, "
                f"got {self.initial_learning_rate}"
            )
        if self.initial_radius is not None and self.initial_radius <= 0:
            raise ValueError(
                f"initial_radius must be positive, got {self.initial_radius}"
            )
        if self.radius_cutoff is not None and self.radius_cutoff <= 0:
            raise ValueError(
                f"radius_cutoff must be positive, got {self.radius_cutoff}"
            )

    def resolved_radius(self, width: int, height: int) -> float:
        """Return the initial radius, defaulting to half the longest side."""
        if self.initial_radius is not None:
            return self.initial_radius
        return max(width, height) / 2
