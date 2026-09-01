"""The vectorised implementation must compute what the original computed.

This is the test the refactor rests on. It converts "I rewrote it and it looks
similar" into "it computes the same thing". If a case here fails, fix the
implementation -- never loosen TOLERANCE.
"""

from __future__ import annotations

import numpy as np
import pytest

from kohonen import SOM, TrainingConfig
from kohonen._reference import reference_train
from kohonen.som import initialise_weights

TOLERANCE = 1e-12


@pytest.mark.parametrize(("width", "height"), [(4, 4), (5, 3), (8, 8)])
@pytest.mark.parametrize("n_features", [3, 10])
@pytest.mark.parametrize("seed", [0, 123])
def test_matches_reference_implementation(
    width: int, height: int, n_features: int, seed: int
) -> None:
    n_iterations = 20
    data = np.random.default_rng(seed + 1000).random((6, n_features))
    expected = reference_train(
        data,
        n_iterations,
        width,
        height,
        initialise_weights(width, height, n_features, seed),
    )
    actual = (
        SOM(
            width,
            height,
            config=TrainingConfig(n_iterations=n_iterations, seed=seed),
        )
        .fit(data)
        .weights_
    )
    np.testing.assert_allclose(actual, expected, atol=TOLERANCE, rtol=0)


def test_matches_reference_on_the_notebooks_own_example() -> None:
    """The 10x10 / 100-iteration configuration from the challenge notebook."""
    data = np.random.default_rng(1000).random((10, 3))
    expected = reference_train(data, 100, 10, 10, initialise_weights(10, 10, 3, 0))
    actual = (
        SOM(10, 10, config=TrainingConfig(n_iterations=100, seed=0)).fit(data).weights_
    )
    np.testing.assert_allclose(actual, expected, atol=TOLERANCE, rtol=0)


def test_deviation_is_within_floating_point_noise() -> None:
    """Report the actual deviation, so drift becomes visible in the failure."""
    data = np.random.default_rng(1003).random((8, 3))
    expected = reference_train(data, 50, 6, 6, initialise_weights(6, 6, 3, 3))
    actual = (
        SOM(6, 6, config=TrainingConfig(n_iterations=50, seed=3)).fit(data).weights_
    )
    deviation = float(np.abs(actual - expected).max())
    assert deviation < TOLERANCE, f"max deviation {deviation:.3e}"
