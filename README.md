# kohonen

A vectorised, tested implementation of the Kohonen Self-Organising Map.

[![CI](https://github.com/wangbenya/kohonen-som/actions/workflows/ci.yml/badge.svg)](https://github.com/wangbenya/kohonen-som/actions/workflows/ci.yml)

**161× faster than the reference implementation, and provably the same
algorithm** — maximum deviation `2.220e-16`, exactly one float64 epsilon.

This repository started as a code review of a single-function SOM
implementation. The review is in **[docs/REVIEW.md](docs/REVIEW.md)**; the
deployment design is in
**[docs/adr/0001-productionisation.md](docs/adr/0001-productionisation.md)**.

## Install

```bash
uv sync --extra dev
```

## Quickstart

```python
import numpy as np
from kohonen import SOM, TrainingConfig

data = np.random.default_rng(0).random((100, 3))

som = SOM(10, 10, config=TrainingConfig(n_iterations=100, seed=42)).fit(data)

labels = som.predict(data)                 # flat BMU index per sample
error = som.quantisation_error(data)       # mean distance to BMU
image = som.weights_                       # (10, 10, 3), plottable as RGB
```

Recover grid coordinates from a flat index with
`np.unravel_index(labels, (som.width, som.height))`.

The API follows scikit-learn's conventions — hyperparameters in `__init__`,
learned state as `weights_`, `fit` returning `self` — **without** taking
scikit-learn as a dependency. The only runtime requirement is NumPy.

## Benchmarks

Python 3.13.14, NumPy 2.5.2, Windows, AMD64. Regenerate with
`uv run --extra dev --extra bench python benchmarks/compare.py`.

| Grid | Original | This library | Speed-up | MiniSom |
|---|---|---|---|---|
| 10×10, 100 iterations | 0.315 s | **0.0133 s** | 24× | 0.0181 s |
| 30×30, 100 iterations | 2.791 s | **0.0299 s** | 93× | 0.0421 s |
| 100×100, 10 iterations | 3.091 s | **0.0192 s** | 161× | 0.0294 s |

The speed-up **scales with grid size**, because what is removed is per-node
Python interpreter overhead rather than algorithmic cost. The original runs at
a flat ~3100 ns per weight update at every size; this library goes from 133 to
19 ns as grids grow and NumPy amortises better.

MiniSom uses asymptotic decay rather than the exponential decay this brief
specifies, so it is not solving quite the same problem — see
[the review](docs/REVIEW.md#on-recommendation-5) for why the benchmark
normalises on weight updates rather than wall-clock.

`somoclu` is supported by the harness but is **not** included in these figures:
it does not build on Windows. The benchmark skips it gracefully.

## Development

```bash
uv run --extra dev pytest          # 103 tests, 100% coverage (gate: 95%)
uv run --extra dev ruff check .
uv run --extra dev mypy            # strict
```

CI runs lint, strict type checking, and the full suite across Python
3.10–3.14 on every branch.

### How the tests are structured

| Suite | Purpose |
|---|---|
| `test_equivalence.py` | Asserts this implementation matches the original to `atol=1e-12` across 38 parametrised cases. The safety net the refactor rests on. |
| `test_properties.py` | Hypothesis-driven invariants: the BMU is the true argmin, decay is monotonic, weights stay in the data envelope, seeds are deterministic. |
| `test_som.py` | Behaviour and edge cases — degenerate grids, zero iterations, non-finite input, dimensionality mismatch. |
| `test_reference.py` | Guards the oracle, including a test that no shipped module imports it. |

`src/kohonen/_reference.py` deliberately preserves the original,
un-optimised implementation. It is the test oracle — see
[the review](docs/REVIEW.md#on-recommendation-4).

## Layout

```
src/kohonen/       the package
tests/             103 tests, 100% coverage
benchmarks/        comparison harness + snapshot exporter
docs/REVIEW.md     the code review and five recommendations
docs/adr/          productionisation design
notebooks/         the original challenge notebook, unchanged
```

## Licence

MIT
