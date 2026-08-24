"""Invariants that must hold for any valid input, not just chosen examples."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from kohonen import SOM, TrainingConfig

SLOW = settings(max_examples=25, deadline=None)

GOLDEN = Path(__file__).parent / "fixtures" / "golden_weights_8x8_seed0.npy"


@SLOW
@given(
    width=st.integers(min_value=3, max_value=8),
    height=st.integers(min_value=3, max_value=8),
    n_features=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=999),
)
def test_bmu_is_the_true_nearest_node(
    width: int, height: int, n_features: int, seed: int
) -> None:
    data = np.random.default_rng(seed).random((5, n_features))
    som = SOM(width, height, config=TrainingConfig(n_iterations=5, seed=seed)).fit(data)

    flat = som.weights_.reshape(som.n_nodes, -1)
    for sample, predicted in zip(data, som.predict(data), strict=True):
        distances = np.linalg.norm(flat - sample, axis=1)
        assert distances[predicted] == distances.min()


@SLOW
@given(seed=st.integers(min_value=0, max_value=999))
def test_identical_seeds_give_identical_maps(seed: int) -> None:
    data = np.random.default_rng(seed).random((6, 3))
    cfg = TrainingConfig(n_iterations=10, seed=seed)
    np.testing.assert_array_equal(
        SOM(5, 5, config=cfg).fit(data).weights_,
        SOM(5, 5, config=cfg).fit(data).weights_,
    )


@SLOW
@given(
    width=st.integers(min_value=3, max_value=7),
    height=st.integers(min_value=3, max_value=7),
    seed=st.integers(min_value=0, max_value=999),
)
def test_weights_stay_within_the_data_envelope(
    width: int, height: int, seed: int
) -> None:
    """Weights move toward data, so they cannot escape the joint bounds.

    Initial weights are drawn from [0, 1) and the data lives in [0, 1), so
    every weight must remain inside the union of those two ranges.
    """
    data = np.random.default_rng(seed).random((8, 3))
    som = SOM(width, height, config=TrainingConfig(n_iterations=20, seed=seed)).fit(
        data
    )

    lower = min(0.0, float(data.min()))
    upper = max(1.0, float(data.max()))
    assert som.weights_.min() >= lower - 1e-12
    assert som.weights_.max() <= upper + 1e-12


@SLOW
@given(
    n_iterations=st.integers(min_value=2, max_value=60),
    seed=st.integers(min_value=0, max_value=999),
)
def test_sigma_and_alpha_decrease_monotonically(n_iterations: int, seed: int) -> None:
    data = np.random.default_rng(seed).random((4, 3))
    sigmas: list[float] = []
    alphas: list[float] = []

    def record(iteration: int, weights: np.ndarray, sigma: float, alpha: float) -> None:
        sigmas.append(sigma)
        alphas.append(alpha)

    SOM(5, 5, config=TrainingConfig(n_iterations=n_iterations, seed=seed)).fit(
        data, on_iteration=record
    )

    assert all(b < a for a, b in pairwise(sigmas))
    assert all(b < a for a, b in pairwise(alphas))


def _golden_weights() -> np.ndarray:
    """Train the configuration the golden fixture pins."""
    data = np.random.default_rng(0).random((10, 3))
    return SOM(8, 8, config=TrainingConfig(n_iterations=50, seed=0)).fit(data).weights_


def test_golden_regression_fixture() -> None:
    """Pin numerical output so unintended drift is caught in CI.

    Regenerate deliberately with ``python tests/regenerate_golden.py`` and
    review the diff -- a change here means the algorithm changed.
    """
    np.testing.assert_allclose(_golden_weights(), np.load(GOLDEN), atol=1e-12, rtol=0)
