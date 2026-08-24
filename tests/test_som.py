from __future__ import annotations

import numpy as np
import pytest

from kohonen import SOM, TrainingConfig
from kohonen.som import initialise_weights


def test_initialise_weights_is_reproducible() -> None:
    a = initialise_weights(4, 3, 5, seed=42)
    b = initialise_weights(4, 3, 5, seed=42)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4, 3, 5)


def test_initialise_weights_differs_across_seeds() -> None:
    a = initialise_weights(4, 3, 5, seed=1)
    b = initialise_weights(4, 3, 5, seed=2)
    assert not np.array_equal(a, b)


def test_fit_returns_self_for_chaining() -> None:
    data = np.random.default_rng(0).random((6, 3))
    som = SOM(4, 4, config=TrainingConfig(n_iterations=5, seed=0))
    assert som.fit(data) is som


def test_fit_infers_dimensionality_from_data() -> None:
    """The original hardcoded 3 features; this must accept any width."""
    data = np.random.default_rng(0).random((6, 7))
    som = SOM(4, 4, config=TrainingConfig(n_iterations=3, seed=0)).fit(data)
    assert som.weights_.shape == (4, 4, 7)


def test_weights_are_unavailable_before_fit() -> None:
    som = SOM(4, 4)
    with pytest.raises(AttributeError, match="not fitted"):
        _ = som.weights_


def test_predict_returns_flat_node_indices() -> None:
    data = np.random.default_rng(0).random((6, 3))
    som = SOM(4, 5, config=TrainingConfig(n_iterations=5, seed=0)).fit(data)
    labels = som.predict(data)
    assert labels.shape == (6,)
    assert labels.min() >= 0
    assert labels.max() < 4 * 5


def test_transform_returns_distance_to_every_node() -> None:
    data = np.random.default_rng(0).random((6, 3))
    som = SOM(4, 5, config=TrainingConfig(n_iterations=5, seed=0)).fit(data)
    assert som.transform(data).shape == (6, 20)


def test_predict_agrees_with_transform() -> None:
    data = np.random.default_rng(0).random((6, 3))
    som = SOM(4, 5, config=TrainingConfig(n_iterations=5, seed=0)).fit(data)
    np.testing.assert_array_equal(som.predict(data), som.transform(data).argmin(axis=1))


def test_quantisation_error_is_non_negative_and_finite() -> None:
    data = np.random.default_rng(0).random((10, 3))
    som = SOM(5, 5, config=TrainingConfig(n_iterations=20, seed=0)).fit(data)
    error = som.quantisation_error(data)
    assert error >= 0
    assert np.isfinite(error)


def test_training_reduces_quantisation_error() -> None:
    """Training must move the map closer to the data.

    The data seed and the weight seed must differ. Sharing one makes both
    draw from the same stream, so every sample lands exactly on a node and
    the untrained error is 0.0 -- which would make this test vacuous.
    """
    data = np.random.default_rng(99).random((30, 3))
    untrained = SOM(6, 6, config=TrainingConfig(n_iterations=0, seed=0)).fit(data)
    trained = SOM(6, 6, config=TrainingConfig(n_iterations=50, seed=0)).fit(data)
    assert trained.quantisation_error(data) < untrained.quantisation_error(data)


def test_zero_iterations_leaves_weights_at_initialisation() -> None:
    data = np.random.default_rng(0).random((6, 3))
    som = SOM(4, 4, config=TrainingConfig(n_iterations=0, seed=7)).fit(data)
    np.testing.assert_array_equal(som.weights_, initialise_weights(4, 4, 3, 7))


@pytest.mark.parametrize(("width", "height"), [(0, 4), (4, 0), (-1, 4)])
def test_invalid_grid_dimensions_are_rejected(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        SOM(width, height)


@pytest.mark.parametrize(
    "bad_data",
    [
        np.array([1.0, 2.0, 3.0]),  # 1-D
        np.zeros((2, 2, 2)),  # 3-D
        np.empty((0, 3)),  # no samples
        np.empty((2, 0)),  # no features
        np.array([[np.nan, 1.0, 2.0]]),  # not finite
        np.array([[np.inf, 1.0, 2.0]]),  # not finite
    ],
)
def test_invalid_training_data_is_rejected(bad_data: np.ndarray) -> None:
    with pytest.raises(ValueError):
        SOM(3, 3, config=TrainingConfig(n_iterations=1, seed=0)).fit(bad_data)


def test_predict_rejects_mismatched_dimensionality() -> None:
    som = SOM(3, 3, config=TrainingConfig(n_iterations=1, seed=0))
    som.fit(np.random.default_rng(0).random((4, 3)))
    with pytest.raises(ValueError, match="features"):
        som.predict(np.random.default_rng(0).random((4, 5)))


def test_fit_does_not_mutate_input_data() -> None:
    data = np.random.default_rng(0).random((6, 3))
    original = data.copy()
    SOM(4, 4, config=TrainingConfig(n_iterations=5, seed=0)).fit(data)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(("width", "height"), [(1, 1), (2, 2)])
def test_degenerate_grids_train_without_error(width: int, height: int) -> None:
    """The original divided by zero here."""
    data = np.random.default_rng(0).random((4, 3))
    with pytest.warns(UserWarning, match="initial radius"):
        som = SOM(width, height, config=TrainingConfig(n_iterations=10, seed=0)).fit(
            data
        )
    assert np.isfinite(som.weights_).all()


def test_on_iteration_callback_receives_every_iteration() -> None:
    data = np.random.default_rng(0).random((4, 3))
    seen: list[tuple[int, float, float]] = []

    def record(iteration: int, weights: np.ndarray, sigma: float, alpha: float) -> None:
        seen.append((iteration, sigma, alpha))

    SOM(4, 4, config=TrainingConfig(n_iterations=7, seed=0)).fit(
        data, on_iteration=record
    )

    assert [entry[0] for entry in seen] == list(range(7))


def test_fit_without_callback_is_unaffected() -> None:
    """The callback must not change the numerical result."""
    data = np.random.default_rng(0).random((5, 3))
    cfg = TrainingConfig(n_iterations=10, seed=0)
    plain = SOM(4, 4, config=cfg).fit(data).weights_
    observed = SOM(4, 4, config=cfg).fit(data, on_iteration=lambda *a: None)
    np.testing.assert_array_equal(plain, observed.weights_)
