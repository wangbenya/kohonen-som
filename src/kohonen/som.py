"""The Kohonen Self-Organising Map."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MIN_RADIUS = 1.0 + 1e-9
"""Floor for the initial radius.

The brief derives ``tau = n_iterations / log(sigma_0)``, which is undefined at
``sigma_0 == 1`` (a 2x2 grid) and negative below it (1x1), where a negative
time constant makes the neighbourhood *grow* instead of decay. Small grids are
legitimate input, so the radius is clamped rather than rejected.
"""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters for a training run.

    Frozen and hashable so a whole run's parameters can be logged in one call
    (MLflow params, for instance) without defensive copying.

    Args:
        n_iterations: Passes over the training data. Zero yields an untrained
            map, which is useful as a baseline.
        initial_learning_rate: Learning rate at t=0.
        initial_radius: Neighbourhood radius at t=0. ``None`` means
            ``max(width, height) / 2``, as the brief specifies.
        seed: Seed for weight initialisation. ``None`` is not reproducible.
    """

    n_iterations: int = 100
    initial_learning_rate: float = 0.1
    initial_radius: float | None = None
    seed: int | None = None

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

    def resolved_radius(self, width: int, height: int) -> float:
        """The initial radius, defaulting to half the longest side."""
        if self.initial_radius is not None:
            return self.initial_radius
        return max(width, height) / 2


def initialise_weights(
    width: int, height: int, n_features: int, seed: int | None
) -> NDArray[np.float64]:
    """Draw uniform random weights in ``[0, 1)`` from a seeded generator."""
    return np.random.default_rng(seed).random((width, height, n_features))


def _validate(data: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            "data must be 2-D with shape (n_samples, n_features), got "
            f"{array.ndim}-D with shape {array.shape}"
        )
    if array.shape[0] == 0:
        raise ValueError("data must contain at least one sample")
    if array.shape[1] == 0:
        raise ValueError("data must contain at least one feature")
    if not np.isfinite(array).all():
        raise ValueError("data must contain only finite values (no NaN or inf)")
    return array


class SOM:
    """A Kohonen Self-Organising Map.

    Follows scikit-learn's conventions -- hyperparameters in ``__init__``,
    learned state suffixed with an underscore, ``fit`` returning ``self`` --
    without taking scikit-learn as a dependency.

    Example:
        >>> import numpy as np
        >>> data = np.random.default_rng(0).random((20, 3))
        >>> som = SOM(10, 10, config=TrainingConfig(n_iterations=100, seed=0))
        >>> labels = som.fit(data).predict(data)
    """

    def __init__(
        self, width: int, height: int, *, config: TrainingConfig | None = None
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be positive, got {width}x{height}")
        self.width = width
        self.height = height
        self.config = config or TrainingConfig()
        self._weights: NDArray[np.float64] | None = None

    @property
    def weights_(self) -> NDArray[np.float64]:
        """Learned node weights, shaped ``(width, height, n_features)``."""
        if self._weights is None:
            raise AttributeError(
                "SOM is not fitted; call fit() before accessing weights_"
            )
        return self._weights

    @property
    def n_nodes(self) -> int:
        return self.width * self.height

    def fit(self, data: NDArray[np.floating]) -> SOM:
        """Train the map. Returns ``self``, for chaining."""
        array = _validate(data)
        cfg = self.config
        weights = initialise_weights(self.width, self.height, array.shape[1], cfg.seed)

        if cfg.n_iterations > 0:
            sigma_0 = cfg.resolved_radius(self.width, self.height)
            if sigma_0 <= MIN_RADIUS:
                warnings.warn(
                    f"initial radius {sigma_0} is too small for a stable time "
                    f"constant (grid is 2x2 or smaller); clamping to "
                    f"{MIN_RADIUS}",
                    UserWarning,
                    stacklevel=2,
                )
                sigma_0 = MIN_RADIUS
            tau = cfg.n_iterations / math.log(sigma_0)

            # Built once, reused for every sample. Walking this with nested
            # Python loops is what made the original implementation slow.
            grid_x, grid_y = np.meshgrid(
                np.arange(self.width), np.arange(self.height), indexing="ij"
            )
            shape = (self.width, self.height)

            for t in range(cfg.n_iterations):
                sigma_t = sigma_0 * math.exp(-t / tau)
                alpha_t = cfg.initial_learning_rate * math.exp(-t / tau)
                two_sigma_sq = 2.0 * sigma_t * sigma_t

                for sample in array:
                    diff = weights - sample
                    # No square root: it does not change an argmin.
                    bmu = int(np.argmin(np.einsum("ijk,ijk->ij", diff, diff)))
                    bmu_x, bmu_y = np.unravel_index(bmu, shape)
                    d2 = (grid_x - bmu_x) ** 2 + (grid_y - bmu_y) ** 2
                    theta = np.exp(-d2 / two_sigma_sq)
                    # w += alpha * theta * (v - w), with diff = w - v
                    weights -= (alpha_t * theta)[..., None] * diff

        self._weights = weights
        return self

    def transform(self, data: NDArray[np.floating]) -> NDArray[np.float64]:
        """Euclidean distance from each sample to every node.

        Returns an array shaped ``(n_samples, width * height)``.
        """
        array = _validate(data)
        expected = self.weights_.shape[2]
        if array.shape[1] != expected:
            raise ValueError(
                f"data has {array.shape[1]} features but the map was fitted "
                f"on {expected} features"
            )
        flat = self.weights_.reshape(self.n_nodes, -1)
        diff = array[:, None, :] - flat[None, :, :]
        distances: NDArray[np.float64] = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        return distances

    def predict(self, data: NDArray[np.floating]) -> NDArray[np.intp]:
        """Flat index of each sample's best matching unit.

        Recover grid coordinates with
        ``np.unravel_index(result, (som.width, som.height))``.
        """
        return np.asarray(self.transform(data).argmin(axis=1), dtype=np.intp)

    def quantisation_error(self, data: NDArray[np.floating]) -> float:
        """Mean distance from each sample to its best matching unit."""
        return float(self.transform(data).min(axis=1).mean())
