"""Regenerate the golden weights fixture. Run deliberately; review the diff.

A change to this fixture means the algorithm's numerical output changed. That
is sometimes intended and sometimes a bug, so regeneration is a separate,
explicit act rather than something a test does for you.

Only regenerate when ``tests/test_equivalence.py`` passes -- a fixture built
from an unverified implementation would pin a bug in place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kohonen import SOM, TrainingConfig

OUTPUT = Path(__file__).parent / "fixtures" / "golden_weights_8x8_seed0.npy"


def main() -> None:
    data = np.random.default_rng(0).random((10, 3))
    weights = (
        SOM(8, 8, config=TrainingConfig(n_iterations=50, seed=0)).fit(data).weights_
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT, weights)
    print(f"Wrote {OUTPUT} with shape {weights.shape}")


if __name__ == "__main__":
    main()
