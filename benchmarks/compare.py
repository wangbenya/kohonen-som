"""Three-way comparison: original, vectorised, and existing packages.

Writes machine-readable JSON to ``benchmarks/results/comparison.json``. Every
figure quoted in the README, the review, or the published site reads from that
file -- no number is ever transcribed by hand.

On fairness: the original and vectorised implementations are the same
algorithm, which the equivalence suite proves to 1e-12. MiniSom is not -- it
uses asymptotic decay rather than the exponential decay this brief specifies.
Comparing raw wall-clock across all three would therefore mislead, so the
headline figure is *time per weight update* and the algorithmic differences
are recorded in each row's ``notes``.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from kohonen import SOM, TrainingConfig
from kohonen._reference import reference_train
from kohonen.som import initialise_weights

RESULTS = Path(__file__).parent / "results" / "comparison.json"

MINISOM_NOTE = (
    "Different algorithm: MiniSom uses asymptotic decay 1/(1+t/(T/2)), not "
    "the exponential decay this brief specifies. Compare seconds_per_update, "
    "not total runtime."
)


@dataclass
class Run:
    implementation: str
    width: int
    height: int
    n_iterations: int
    n_samples: int
    n_features: int
    seconds: float
    weight_updates: int
    seconds_per_update: float
    quantisation_error: float
    notes: str = ""


def _quantisation_error(
    weights: NDArray[np.float64], data: NDArray[np.float64]
) -> float:
    flat = weights.reshape(-1, weights.shape[-1])
    diff = data[:, None, :] - flat[None, :, :]
    return float(np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)).min(1).mean())


def _make_run(
    name: str,
    seconds: float,
    weights: NDArray[np.float64],
    data: NDArray[np.float64],
    width: int,
    height: int,
    n_iterations: int,
    notes: str = "",
) -> Run:
    updates = n_iterations * data.shape[0] * width * height
    return Run(
        implementation=name,
        width=width,
        height=height,
        n_iterations=n_iterations,
        n_samples=int(data.shape[0]),
        n_features=int(data.shape[1]),
        seconds=seconds,
        weight_updates=updates,
        seconds_per_update=seconds / updates if updates else 0.0,
        quantisation_error=_quantisation_error(weights, data),
        notes=notes,
    )


def bench_original(
    data: NDArray[np.float64], width: int, height: int, n_iterations: int
) -> Run | None:
    w0 = initialise_weights(width, height, data.shape[1], 0)
    start = time.perf_counter()
    weights = reference_train(data, n_iterations, width, height, w0)
    elapsed = time.perf_counter() - start
    return _make_run(
        "original",
        elapsed,
        weights,
        data,
        width,
        height,
        n_iterations,
        "Triple-nested Python loop, as written in the challenge notebook.",
    )


def bench_vectorised(
    data: NDArray[np.float64], width: int, height: int, n_iterations: int
) -> Run | None:
    som = SOM(width, height, config=TrainingConfig(n_iterations=n_iterations, seed=0))
    start = time.perf_counter()
    som.fit(data)
    elapsed = time.perf_counter() - start
    return _make_run(
        "vectorised",
        elapsed,
        som.weights_,
        data,
        width,
        height,
        n_iterations,
        "Same algorithm as 'original'; equivalence asserted to 1e-12 in CI.",
    )


def bench_minisom(
    data: NDArray[np.float64], width: int, height: int, n_iterations: int
) -> Run | None:
    try:
        from minisom import MiniSom
    except ImportError:
        return None
    som = MiniSom(
        width,
        height,
        data.shape[1],
        sigma=max(width, height) / 2,
        learning_rate=0.1,
        random_seed=0,
    )
    start = time.perf_counter()
    som.train(data, n_iterations * data.shape[0], use_epochs=False)
    elapsed = time.perf_counter() - start
    return _make_run(
        "minisom",
        elapsed,
        np.asarray(som.get_weights()),
        data,
        width,
        height,
        n_iterations,
        MINISOM_NOTE,
    )


def bench_somoclu(
    data: NDArray[np.float64], width: int, height: int, n_iterations: int
) -> Run | None:
    try:
        import somoclu
    except ImportError:
        return None
    som = somoclu.Somoclu(height, width, initialization="random", verbose=0)
    start = time.perf_counter()
    som.train(data.astype(np.float32), epochs=n_iterations)
    elapsed = time.perf_counter() - start
    weights = np.asarray(som.codebook, dtype=np.float64).transpose(1, 0, 2)
    return _make_run(
        "somoclu",
        elapsed,
        weights,
        data,
        width,
        height,
        n_iterations,
        "C++/OpenMP, batch SOM. A different algorithm and parallelised; "
        "included to show where an existing library beats this one.",
    )


BenchFn = Callable[[NDArray[np.float64], int, int, int], "Run | None"]

BENCHMARKS: list[tuple[str, BenchFn]] = [
    ("original", bench_original),
    ("vectorised", bench_vectorised),
    ("minisom", bench_minisom),
    ("somoclu", bench_somoclu),
]


def main() -> None:
    configurations = [(10, 10, 100), (30, 30, 100), (100, 100, 10)]
    rng = np.random.default_rng(0)
    runs: list[Run] = []

    for width, height, n_iterations in configurations:
        data = rng.random((10, 3))
        print(f"--- {width}x{height}, {n_iterations} iterations ---")
        for name, fn in BENCHMARKS:
            run = fn(data, width, height, n_iterations)
            if run is None:
                print(f"  {name:12s} SKIPPED (not installed)")
                continue
            runs.append(run)
            print(
                f"  {run.implementation:12s} {run.seconds:8.3f}s"
                f"  ({run.seconds_per_update * 1e9:7.1f} ns/update)"
                f"  qe={run.quantisation_error:.4f}"
            )

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "platform": {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "system": platform.system(),
                    "processor": platform.processor(),
                },
                "runs": [asdict(r) for r in runs],
            },
            indent=2,
        )
    )
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    main()
