"""Export training snapshots for the demo animation on the site.

The animation is driven by this file, so what a viewer watches is the tested
library's real behaviour rather than a JavaScript reimplementation.

Weights are quantised to 8-bit integers. They are rendered as RGB, so the page
cannot display more precision than that, and integers keep the payload roughly
an order of magnitude smaller than full floats.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kohonen import SOM, TrainingConfig

OUTPUT = Path(__file__).parent / "results" / "snapshots.json"

WIDTH = 16
HEIGHT = 16
N_ITERATIONS = 120
N_FRAMES = 40


def main() -> None:
    rng = np.random.default_rng(0)
    data = rng.random((12, 3))

    # Frames are sampled on a curve rather than evenly: the map changes fastest
    # early on, so linear sampling would spend most frames on a nearly static
    # image.
    checkpoints = sorted(
        {
            round((i / (N_FRAMES - 1)) ** 1.7 * (N_ITERATIONS - 1))
            for i in range(N_FRAMES)
        }
    )

    frames: list[dict[str, object]] = []
    for stop in checkpoints:
        som = SOM(
            WIDTH,
            HEIGHT,
            config=TrainingConfig(n_iterations=stop + 1, seed=0),
        ).fit(data)
        quantised = np.clip(np.round(som.weights_ * 255), 0, 255).astype(np.uint8)
        frames.append({"iteration": stop + 1, "rgb": quantised.ravel().tolist()})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "width": WIDTH,
                "height": HEIGHT,
                "n_iterations": N_ITERATIONS,
                "colours": np.clip(np.round(data * 255), 0, 255)
                .astype(np.uint8)
                .tolist(),
                "frames": frames,
            },
            separators=(",", ":"),
        )
    )
    print(
        f"Wrote {OUTPUT} ({len(frames)} frames, {OUTPUT.stat().st_size / 1024:.0f} KB)"
    )


if __name__ == "__main__":
    main()
