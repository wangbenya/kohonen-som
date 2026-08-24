"""The Kohonen Self-Organising Map estimator."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from kohonen.config import TrainingConfig
from kohonen.decay import ExponentialDecay, clamped_radius, time_constant
from kohonen.neighbourhood import (
    coordinate_grid,
    gaussian_influence,
    squared_distance_to,
)


class IterationCallback(Protocol):
    """Observer invoked once per training iteration.

    Implementations must treat ``weights`` as read-only; it is the live
    training array, not a copy. Copy it if you need to retain it.
    """

    def __call__(
        self,
        iteration: int,
        weights: NDArray[np.float64],
        sigma: float,
        alpha: float,
    ) -> None: ...


def initialise_weights(
    width: int, height: int, n_features: int, seed: int | None
) -> NDArray[np.float64]:
    """Draw uniform random weights in ``[0, 1)`` from a seeded generator."""
    rng = np.random.default_rng(seed)
    return rng.random((width, height, n_features))


def _validate_data(data: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"data must be 2-D with shape (n_samples, n_features), "
            f"got {array.ndim}-D with shape {array.shape}"
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
    without taking a dependency on scikit-learn.

    Example:
        >>> import numpy as np
        >>> data = np.random.default_rng(0).random((20, 3))
        >>> som = SOM(10, 10, config=TrainingConfig(n_iterations=100, seed=0))
        >>> labels = som.fit(data).predict(data)
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        config: TrainingConfig | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be positive, got {width}x{height}")
        self.width = width
        self.height = height
        self.config = config if config is not None else TrainingConfig()
        self._weights: NDArray[np.float64] | None = None

    @property
    def weights_(self) -> NDArray[np.float64]:
        """The learned node weights, shaped ``(width, height, n_features)``."""
        if self._weights is None:
            raise AttributeError(
                "SOM is not fitted; call fit() before accessing weights_"
            )
        return self._weights

    @property
    def n_nodes(self) -> int:
        """Total node count, ``width * height``."""
        return self.width * self.height

    def fit(
        self,
        data: NDArray[np.floating],
        *,
        on_iteration: IterationCallback | None = None,
    ) -> SOM:
        """Train the map.

        Args:
            data: Training vectors, shaped ``(n_samples, n_features)``.
            on_iteration: Optional observer called once per iteration. When
                ``None`` (the default) there is no overhead.

        Returns:
            ``self``, for chaining.
        """
        array = _validate_data(data)
        n_features = array.shape[1]
        cfg = self.config

        weights = initialise_weights(self.width, self.height, n_features, cfg.seed)

        if cfg.n_iterations > 0:
            sigma_0 = cfg.resolved_radius(self.width, self.height)
            tau = time_constant(cfg.n_iterations, sigma_0)
            sigma_at = ExponentialDecay(clamped_radius(sigma_0), tau)
            alpha_at = ExponentialDecay(cfg.initial_learning_rate, tau)
            grid_x, grid_y = coordinate_grid(self.width, self.height)
            grid_shape = (self.width, self.height)

            for t in range(cfg.n_iterations):
                sigma_t = sigma_at(t)
                alpha_t = alpha_at(t)
                for sample in array:
                    diff = weights - sample
                    flat_bmu = int(np.argmin(np.einsum("ijk,ijk->ij", diff, diff)))
                    bmu_x, bmu_y = np.unravel_index(flat_bmu, grid_shape)
                    d2 = squared_distance_to(grid_x, grid_y, int(bmu_x), int(bmu_y))
                    theta = gaussian_influence(d2, sigma_t)
                    weights -= (alpha_t * theta)[..., None] * diff
                if on_iteration is not None:
                    on_iteration(t, weights, sigma_t, alpha_t)

        self._weights = weights
        return self

    def transform(self, data: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return Euclidean distances from each sample to every node.

        Returns:
            Array shaped ``(n_samples, width * height)``.
        """
        array = self._validate_for_inference(data)
        flat = self.weights_.reshape(self.n_nodes, -1)
        diff = array[:, None, :] - flat[None, :, :]
        distances: NDArray[np.float64] = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        return distances

    def predict(self, data: NDArray[np.floating]) -> NDArray[np.intp]:
        """Return the flat index of each sample's best matching unit.

        Recover grid coordinates with
        ``np.unravel_index(result, (som.width, som.height))``.
        """
        return np.asarray(self.transform(data).argmin(axis=1), dtype=np.intp)

    def quantisation_error(self, data: NDArray[np.floating]) -> float:
        """Mean distance from each sample to its best matching unit."""
        return float(self.transform(data).min(axis=1).mean())

    def _validate_for_inference(
        self, data: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        array = _validate_data(data)
        expected = self.weights_.shape[2]
        if array.shape[1] != expected:
            raise ValueError(
                f"data has {array.shape[1]} features but the map was fitted "
                f"on {expected} features"
            )
        return array
