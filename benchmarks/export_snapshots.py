"""Export real training snapshots for the executive explainer animation.

The published animation is driven by this file, so what viewers watch is the
tested library's actual behaviour rather than a JavaScript reimplementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kohonen import SOM, TrainingConfig

OUTPUT = Path(__file__).parent / "results" / "snapshots.json"

WIDTH = 20
HEIGHT = 20
N_ITERATIONS = 200
CAPTURE_EVERY = 2
DECIMALS = 4


def main() -> None:
    rng = np.random.default_rng(0)
    data = rng.random((15, 3))
    frames: list[dict[str, object]] = []

    def capture(
        iteration: int,
        weights: NDArray[np.float64],
        sigma: float,
        alpha: float,
    ) -> None:
        if iteration % CAPTURE_EVERY and iteration != N_ITERATIONS - 1:
            return
        frames.append(
            {
                "iteration": iteration,
                "sigma": round(sigma, 4),
                "alpha": round(alpha, 6),
                "weights": np.round(weights, DECIMALS).ravel().tolist(),
            }
        )

    som = SOM(
        WIDTH,
        HEIGHT,
        config=TrainingConfig(n_iterations=N_ITERATIONS, seed=0),
    )
    som.fit(data, on_iteration=capture)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "width": WIDTH,
                "height": HEIGHT,
                "n_features": 3,
                "n_iterations": N_ITERATIONS,
                "n_samples": int(data.shape[0]),
                "training_data": np.round(data, DECIMALS).tolist(),
                "frames": frames,
            },
            separators=(",", ":"),
        )
    )
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote {OUTPUT} ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
