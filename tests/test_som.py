from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from kohonen import SOM, TrainingConfig
from kohonen.som import initialise_weights

# ---------------------------------------------------------------- config


def test_defaults_match_the_brief() -> None:
    cfg = TrainingConfig()
    assert cfg.n_iterations == 100
    assert cfg.initial_learning_rate == 0.1
    assert cfg.initial_radius is None
    assert cfg.seed is None


def test_config_is_hashable_for_experiment_logging() -> None:
    assert hash(TrainingConfig()) == hash(TrainingConfig())


def test_radius_defaults_to_half_the_longest_side() -> None:
    assert TrainingConfig().resolved_radius(10, 4) == 5.0


def test_radius_honours_explicit_override() -> None:
    assert TrainingConfig(initial_radius=2.5).resolved_radius(10, 4) == 2.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_iterations": -1}, "n_iterations"),
        ({"initial_learning_rate": 0.0}, "initial_learning_rate"),
        ({"initial_learning_rate": -0.5}, "initial_learning_rate"),
        ({"initial_radius": 0.0}, "initial_radius"),
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs)


# ------------------------------------------------------------ weight init


def test_initialisation_is_reproducible() -> None:
    a = initialise_weights(4, 3, 5, seed=42)
    b = initialise_weights(4, 3, 5, seed=42)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4, 3, 5)


def test_initialisation_differs_across_seeds() -> None:
    a = initialise_weights(4, 3, 5, seed=1)
    b = initialise_weights(4, 3, 5, seed=2)
    assert not np.array_equal(a, b)


# ------------------------------------------------------------------- fit


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
    with pytest.raises(AttributeError, match="not fitted"):
        _ = SOM(4, 4).weights_


def test_fit_does_not_mutate_input_data() -> None:
    data = np.random.default_rng(0).random((6, 3))
    original = data.copy()
    SOM(4, 4, config=TrainingConfig(n_iterations=5, seed=0)).fit(data)
    np.testing.assert_array_equal(data, original)


def test_zero_iterations_leaves_weights_at_initialisation() -> None:
    data = np.random.default_rng(0).random((6, 3))
    som = SOM(4, 4, config=TrainingConfig(n_iterations=0, seed=7)).fit(data)
    np.testing.assert_array_equal(som.weights_, initialise_weights(4, 4, 3, 7))


def test_identical_seeds_give_identical_maps() -> None:
    data = np.random.default_rng(5).random((6, 3))
    cfg = TrainingConfig(n_iterations=10, seed=3)
    np.testing.assert_array_equal(
        SOM(5, 5, config=cfg).fit(data).weights_,
        SOM(5, 5, config=cfg).fit(data).weights_,
    )


@pytest.mark.parametrize(("width", "height"), [(1, 1), (2, 2)])
def test_degenerate_grids_train_without_error(width: int, height: int) -> None:
    """The original divided by zero here."""
    data = np.random.default_rng(0).random((4, 3))
    with pytest.warns(UserWarning, match="initial radius"):
        som = SOM(width, height, config=TrainingConfig(n_iterations=10, seed=0)).fit(
            data
        )
    assert np.isfinite(som.weights_).all()


def test_training_reduces_quantisation_error() -> None:
    """Training must move the map closer to the data.

    The data seed and the weight seed must differ. Sharing one makes both draw
    from the same stream, so every sample lands exactly on a node and the
    untrained error is 0.0 -- which would make this test vacuous.
    """
    data = np.random.default_rng(99).random((30, 3))
    untrained = SOM(6, 6, config=TrainingConfig(n_iterations=0, seed=0)).fit(data)
    trained = SOM(6, 6, config=TrainingConfig(n_iterations=50, seed=0)).fit(data)
    assert trained.quantisation_error(data) < untrained.quantisation_error(data)


def test_weights_stay_within_the_data_envelope() -> None:
    """Weights move toward data, so they cannot escape the joint bounds."""
    data = np.random.default_rng(4).random((8, 3))
    som = SOM(6, 6, config=TrainingConfig(n_iterations=30, seed=1)).fit(data)
    assert som.weights_.min() >= -1e-12
    assert som.weights_.max() <= 1.0 + 1e-12


# --------------------------------------------------------------- inference


def test_predict_returns_flat_node_indices() -> None:
    data = np.random.default_rng(0).random((6, 3))
    som = SOM(4, 5, config=TrainingConfig(n_iterations=5, seed=0)).fit(data)
    labels = som.predict(data)
    assert labels.shape == (6,)
    assert labels.min() >= 0
    assert labels.max() < 20


def test_predict_returns_the_true_nearest_node() -> None:
    data = np.random.default_rng(7).random((6, 3))
    som = SOM(5, 4, config=TrainingConfig(n_iterations=10, seed=2)).fit(data)
    flat = som.weights_.reshape(som.n_nodes, -1)
    for sample, predicted in zip(data, som.predict(data), strict=True):
        distances = np.linalg.norm(flat - sample, axis=1)
        assert distances[predicted] == distances.min()


def test_transform_returns_distance_to_every_node() -> None:
    data = np.random.default_rng(0).random((6, 3))
    som = SOM(4, 5, config=TrainingConfig(n_iterations=5, seed=0)).fit(data)
    assert som.transform(data).shape == (6, 20)


def test_quantisation_error_is_non_negative_and_finite() -> None:
    data = np.random.default_rng(0).random((10, 3))
    som = SOM(5, 5, config=TrainingConfig(n_iterations=20, seed=0)).fit(data)
    error = som.quantisation_error(data)
    assert error >= 0
    assert np.isfinite(error)


# -------------------------------------------------------------- validation


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
def test_invalid_data_is_rejected(bad_data: np.ndarray) -> None:
    with pytest.raises(ValueError):
        SOM(3, 3, config=TrainingConfig(n_iterations=1, seed=0)).fit(bad_data)


def test_predict_rejects_mismatched_dimensionality() -> None:
    som = SOM(3, 3, config=TrainingConfig(n_iterations=1, seed=0))
    som.fit(np.random.default_rng(0).random((4, 3)))
    with pytest.raises(ValueError, match="features"):
        som.predict(np.random.default_rng(0).random((4, 5)))
