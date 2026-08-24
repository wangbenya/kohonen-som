"""The original implementation, frozen as an executable test oracle.

This is the code from ``notebooks/kohonen.ipynb``, preserved so the test suite
can prove the vectorised implementation computes the same thing. It is
deliberately NOT optimised and must never be imported by other modules in
``src/`` -- ``tests/test_reference.py`` enforces that isolation.

Two changes were made to the original, and only these two:

1. Initial weights are injected rather than drawn from ``np.random.random``.
   Equivalence is untestable if the two implementations cannot start from
   identical weights.
2. Greek identifiers were transliterated to ASCII (``sigma_0``, ``alpha_t``).

The arithmetic is otherwise character-for-character the original, including
the redundant ``sqrt`` that is immediately squared -- removing it would change
the floating-point result and defeat the purpose of an oracle.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def reference_train(
    input_data: NDArray[np.float64],
    n_max_iterations: int,
    width: int,
    height: int,
    initial_weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Train a SOM using the original triple-nested-loop implementation."""
    sigma_0 = max(width, height) / 2
    alpha_0 = 0.1
    weights = np.array(initial_weights, dtype=np.float64, copy=True)
    lambda_ = n_max_iterations / np.log(sigma_0)
    for t in range(n_max_iterations):
        sigma_t = sigma_0 * np.exp(-t / lambda_)
        alpha_t = alpha_0 * np.exp(-t / lambda_)
        for vt in input_data:
            bmu = np.argmin(np.sum((weights - vt) ** 2, axis=2))
            bmu_x, bmu_y = np.unravel_index(bmu, (width, height))
            for x in range(width):
                for y in range(height):
                    di = np.sqrt(((x - bmu_x) ** 2) + ((y - bmu_y) ** 2))
                    theta_t = np.exp(-(di**2) / (2 * (sigma_t**2)))
                    weights[x, y] += alpha_t * theta_t * (vt - weights[x, y])
    return weights
