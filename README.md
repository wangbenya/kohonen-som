# kohonen

A vectorised, tested implementation of the Kohonen Self-Organising Map.

[![CI](https://github.com/wangbenya/kohonen-som/actions/workflows/ci.yml/badge.svg)](https://github.com/wangbenya/kohonen-som/actions/workflows/ci.yml)
[![Pages](https://github.com/wangbenya/kohonen-som/actions/workflows/pages.yml/badge.svg)](https://wangbenya.github.io/kohonen-som/)

### 📊 [**Read the walkthrough →**](https://wangbenya.github.io/kohonen-som/)

What a SOM is, where the specification contradicts itself, the measured results with a
live demo, and how to reuse the package. Built from this repository on every push.

---

**155× faster than the original implementation, and provably the same algorithm** —
maximum deviation `2.220e-16`, one float64 epsilon.

This repository began as a code review of a single-function SOM. The review, including
the productionisation design, is in **[docs/REVIEW.md](docs/REVIEW.md)**.

## Install

**As a dependency, straight from the repository:**

```bash
uv add git+https://github.com/wangbenya/kohonen-som.git
# or
pip install git+https://github.com/wangbenya/kohonen-som.git
```

**To work on it:**

```bash
git clone https://github.com/wangbenya/kohonen-som.git
cd kohonen-som
uv sync --extra dev
```

**As a wheel, to hand to another team or an air-gapped environment:**

```bash
uv build
pip install dist/kohonen_som-0.1.0-py3-none-any.whl
```

Inside an organisation, publish to a private index — Azure Artifacts, AWS CodeArtifact,
or an internal devpi — rather than public PyPI, so versions stay pinned, auditable, and
inside the tenancy.

## Usage

```python
import numpy as np
from kohonen import SOM, TrainingConfig

data = np.random.default_rng(0).random((100, 3))

som = SOM(10, 10, config=TrainingConfig(n_iterations=100, seed=42)).fit(data)

labels = som.predict(data)  # flat BMU index per sample
error = som.quantisation_error(data)  # mean distance to BMU
image = som.weights_  # (10, 10, 3), plottable as RGB
```

Recover grid coordinates with `np.unravel_index(labels, (som.width, som.height))`.

The API follows scikit-learn's conventions — hyperparameters in `__init__`, learned state
as `weights_`, `fit` returning `self` — without taking scikit-learn as a dependency. The
only runtime requirement is NumPy.

## Benchmarks

Python 3.13.14, NumPy 2.5.2, Windows. Regenerate with
`uv run --extra dev --extra bench python benchmarks/compare.py`.

| Grid | Original | This library | Speed-up | MiniSom |
|---|---|---|---|---|
| 10×10, 100 iterations | 0.308 s | **0.0130 s** | 24× | 0.0176 s |
| 30×30, 100 iterations | 2.750 s | **0.0299 s** | 92× | 0.0417 s |
| 100×100, 10 iterations | 3.034 s | **0.0196 s** | 155× | 0.0295 s |

The speed-up **scales with grid size**, because what is removed is per-node Python
interpreter overhead rather than algorithmic cost: the original runs at a flat ~3050 ns
per weight update at every size, while this library goes from 130 to 20 ns as grids grow.

MiniSom uses asymptotic decay rather than the exponential decay this brief specifies, so
it is not solving quite the same problem — see
[the review](docs/REVIEW.md#on-recommendation-5).

## Development

```bash
uv run --extra dev pytest      # 52 tests, 100% coverage (gate: 95%)
uv run --extra dev ruff check .
uv run --extra dev mypy        # strict
```

CI runs lint, strict type checking, and the full suite across Python 3.10–3.14.

`src/kohonen/_reference.py` deliberately preserves the original, un-optimised
implementation. It is the test oracle: `tests/test_equivalence.py` asserts this library
matches it to `atol=1e-12`, which is what makes the rewrite safe rather than merely
plausible.

## Layout

```
src/kohonen/som.py         the implementation
src/kohonen/_reference.py  the original, kept as a test oracle
tests/                     52 tests
benchmarks/compare.py      original vs this library vs MiniSom
docs/REVIEW.md             the review, recommendations, and productionisation
notebooks/                 the original challenge notebook, unchanged
```

## Licence

MIT
