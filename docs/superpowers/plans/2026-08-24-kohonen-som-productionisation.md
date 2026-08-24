# Kohonen SOM Productionisation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the single-function SOM in `notebooks/kohonen.ipynb` into a tested, typed, benchmarked, pip-installable Python package with a published three-tab explainer site.

**Architecture:** A `src/`-layout package whose training loop is vectorised over a precomputed coordinate grid, preserving the original's online (per-sample) update semantics so it stays bit-comparable to the original. The original implementation is retained as `_reference.py` and used as an executable test oracle. Decay is an injectable policy; training observability is an optional callback with zero cost when unused.

**Tech Stack:** Python 3.10–3.13, NumPy, `uv`, pytest + pytest-cov + hypothesis, ruff, mypy strict, GitHub Actions, MiniSom (benchmark comparator), somoclu (optional comparator).

**Spec:** `docs/superpowers/specs/2026-08-18-kohonen-som-productionisation-design.md`

## Global Constraints

- Python floor **3.10**; CI matrix is 3.10, 3.11, 3.12, 3.13.
- Use `from __future__ import annotations` in every module (keeps `X | None` syntax valid on 3.10).
- **No `scikit-learn` dependency.** Follow its conventions only.
- **No GPU/numba backend.** Explicit non-goal.
- Runtime dependency is **NumPy only**. Everything else is a dev/bench extra.
- `_reference.py` must never be imported by another module in `src/`. A test enforces this.
- Coverage gate `--cov-fail-under=95`.
- `ruff check`, `ruff format --check`, `mypy --strict` must pass clean.
- **No figure is ever transcribed by hand.** Every number in README/REVIEW/site reads from `benchmarks/results/*.json`.
- ASCII identifiers only — no Greek characters in code.
- British spelling in prose (`vectorised`, `quantisation`, `neighbourhood`).
- All commands run through `uv run`. There is no system Python on PATH.
- Commit after every task. Atomic, well-messaged commits are part of the deliverable.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/kohonen/__init__.py`, `src/kohonen/py.typed`, `tests/test_packaging.py`

**Interfaces:**
- Consumes: nothing
- Produces: importable `kohonen` package exposing `__version__: str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_packaging.py
from __future__ import annotations

import kohonen


def test_package_exposes_version() -> None:
    assert isinstance(kohonen.__version__, str)
    assert kohonen.__version__.count(".") >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_packaging.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kohonen'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "kohonen"
version = "0.1.0"
description = "A vectorised, tested Kohonen Self-Organising Map."
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[project.optional-dependencies]
dev = [
  "pytest>=8.0", "pytest-cov>=5.0", "hypothesis>=6.100",
  "ruff>=0.6", "mypy>=1.11",
]
bench = ["minisom>=2.3", "pytest-benchmark>=4.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/kohonen"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=kohonen --cov-report=term-missing --cov-fail-under=95"

[tool.coverage.run]
source = ["kohonen"]
omit = ["*/_reference.py"]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "RUF"]

[tool.mypy]
strict = true
files = ["src", "tests"]

[[tool.mypy.overrides]]
module = ["minisom.*", "somoclu.*"]
ignore_missing_imports = true
```

Note: `_reference.py` is omitted from coverage — it is a frozen oracle, not
maintained code, and holding it to the 95% gate would be meaningless.

```python
# src/kohonen/__init__.py
"""A vectorised, tested implementation of the Kohonen Self-Organising Map."""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
```

```
# src/kohonen/py.typed
```
(empty file)

```gitignore
# .gitignore
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
dist/
build/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
.ipynb_checkpoints/
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_packaging.py -v --no-cov`
Expected: PASS

- [ ] **Step 5: Verify lint and types are clean**

Run: `uv run --extra dev ruff check . && uv run --extra dev ruff format --check . && uv run --extra dev mypy`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src tests uv.lock
git commit -m "Add package scaffold with uv, ruff, mypy strict, and pytest"
```

---

### Task 2: Freeze the original as a test oracle

**Files:**
- Create: `src/kohonen/_reference.py`, `tests/test_reference.py`

**Interfaces:**
- Consumes: nothing
- Produces: `reference_train(input_data: NDArray[np.float64], n_max_iterations: int, width: int, height: int, initial_weights: NDArray[np.float64]) -> NDArray[np.float64]`

**Design note:** the algorithm is copied verbatim from the notebook. The *only*
modification is that initial weights are injected rather than drawn from global
RNG state — without that, equivalence cannot be tested at all. Greek
identifiers are transliterated to ASCII per the global constraints. Both
changes are recorded in the module docstring.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reference.py
from __future__ import annotations

import numpy as np

from kohonen._reference import reference_train


def test_reference_is_deterministic_given_initial_weights() -> None:
    rng = np.random.default_rng(0)
    data = rng.random((5, 3))
    w0 = rng.random((4, 4, 3))

    a = reference_train(data, 10, 4, 4, w0.copy())
    b = reference_train(data, 10, 4, 4, w0.copy())

    np.testing.assert_array_equal(a, b)


def test_reference_does_not_mutate_caller_weights() -> None:
    rng = np.random.default_rng(1)
    data = rng.random((3, 3))
    w0 = rng.random((4, 4, 3))
    original = w0.copy()

    reference_train(data, 5, 4, 4, w0)

    np.testing.assert_array_equal(w0, original)


def test_reference_shape_matches_grid() -> None:
    rng = np.random.default_rng(2)
    data = rng.random((3, 5))
    w0 = rng.random((6, 4, 5))

    out = reference_train(data, 3, 6, 4, w0)

    assert out.shape == (6, 4, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_reference.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'kohonen._reference'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/kohonen/_reference.py
"""The original implementation, frozen as an executable test oracle.

This is Sam's code from ``notebooks/kohonen.ipynb``, preserved so the test
suite can prove the vectorised implementation computes the same thing. It is
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_reference.py -v --no-cov`
Expected: PASS (3 tests)

- [ ] **Step 5: Add the isolation test**

```python
# append to tests/test_reference.py
from pathlib import Path

import kohonen


def test_reference_is_not_imported_by_package_modules() -> None:
    """The oracle must stay quarantined from shipped code."""
    package_dir = Path(kohonen.__file__).parent
    offenders = [
        path.name
        for path in package_dir.glob("*.py")
        if path.name != "_reference.py" and "_reference" in path.read_text()
    ]
    assert offenders == [], f"_reference is imported by: {offenders}"
```

- [ ] **Step 6: Run and commit**

Run: `uv run --extra dev pytest tests/test_reference.py -v --no-cov`
Expected: PASS (4 tests)

```bash
git add src/kohonen/_reference.py tests/test_reference.py
git commit -m "Freeze original implementation as an executable test oracle"
```

---

### Task 3: TrainingConfig

**Files:**
- Create: `src/kohonen/config.py`, `tests/test_config.py`
- Modify: `src/kohonen/__init__.py`

**Interfaces:**
- Consumes: nothing
- Produces: `TrainingConfig(n_iterations: int = 100, initial_learning_rate: float = 0.1, initial_radius: float | None = None, seed: int | None = None, radius_cutoff: float | None = None)` — frozen, slotted, validating in `__post_init__`. Method `resolved_radius(width: int, height: int) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from __future__ import annotations

import dataclasses

import pytest

from kohonen import TrainingConfig


def test_defaults_match_the_brief() -> None:
    cfg = TrainingConfig()
    assert cfg.n_iterations == 100
    assert cfg.initial_learning_rate == 0.1
    assert cfg.initial_radius is None
    assert cfg.seed is None
    assert cfg.radius_cutoff is None


def test_config_is_frozen() -> None:
    cfg = TrainingConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n_iterations = 5  # type: ignore[misc]


def test_config_is_hashable_for_experiment_logging() -> None:
    assert hash(TrainingConfig()) == hash(TrainingConfig())


def test_resolved_radius_defaults_to_half_the_longest_side() -> None:
    assert TrainingConfig().resolved_radius(10, 4) == 5.0


def test_resolved_radius_honours_explicit_override() -> None:
    assert TrainingConfig(initial_radius=2.5).resolved_radius(10, 4) == 2.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_iterations": -1}, "n_iterations"),
        ({"initial_learning_rate": 0.0}, "initial_learning_rate"),
        ({"initial_learning_rate": -0.5}, "initial_learning_rate"),
        ({"initial_radius": 0.0}, "initial_radius"),
        ({"radius_cutoff": 0.0}, "radius_cutoff"),
    ],
)
def test_invalid_values_are_rejected(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs)


def test_zero_iterations_is_permitted() -> None:
    assert TrainingConfig(n_iterations=0).n_iterations == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_config.py -v --no-cov`
Expected: FAIL — `ImportError: cannot import name 'TrainingConfig'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/kohonen/config.py
"""Training hyperparameters, validated at construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters for a SOM training run.

    Frozen and hashable so it can be logged wholesale as experiment
    parameters (for example, MLflow params) without defensive copying.

    Args:
        n_iterations: Number of passes over the training data. Zero is valid
            and yields an untrained map.
        initial_learning_rate: Learning rate at t=0.
        initial_radius: Neighbourhood radius at t=0. When ``None``, defaults
            to ``max(width, height) / 2`` as the brief specifies.
        seed: Seed for weight initialisation. ``None`` means non-reproducible.
        radius_cutoff: When set, restricts each update to a window of
            ``radius_cutoff * sigma_t`` nodes around the BMU. This is an
            approximation, so it is off by default; see ``docs/REVIEW.md``.
    """

    n_iterations: int = 100
    initial_learning_rate: float = 0.1
    initial_radius: float | None = None
    seed: int | None = None
    radius_cutoff: float | None = None

    def __post_init__(self) -> None:
        if self.n_iterations < 0:
            raise ValueError(
                f"n_iterations must be non-negative, got {self.n_iterations}"
            )
        if self.initial_learning_rate <= 0:
            raise ValueError(
                "initial_learning_rate must be positive, "
                f"got {self.initial_learning_rate}"
            )
        if self.initial_radius is not None and self.initial_radius <= 0:
            raise ValueError(
                f"initial_radius must be positive, got {self.initial_radius}"
            )
        if self.radius_cutoff is not None and self.radius_cutoff <= 0:
            raise ValueError(
                f"radius_cutoff must be positive, got {self.radius_cutoff}"
            )

    def resolved_radius(self, width: int, height: int) -> float:
        """Return the initial radius, defaulting to half the longest side."""
        if self.initial_radius is not None:
            return self.initial_radius
        return max(width, height) / 2
```

```python
# src/kohonen/__init__.py  -- replace the __all__ block
from __future__ import annotations

from kohonen.config import TrainingConfig

__version__ = "0.1.0"

__all__ = ["TrainingConfig", "__version__"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_config.py -v --no-cov`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/kohonen/config.py src/kohonen/__init__.py tests/test_config.py
git commit -m "Add validated, frozen TrainingConfig"
```

---

### Task 4: Decay schedules

**Files:**
- Create: `src/kohonen/decay.py`, `tests/test_decay.py`
- Modify: `src/kohonen/__init__.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `DecaySchedule` Protocol with `__call__(self, t: int) -> float`
  - `ExponentialDecay(initial_value: float, time_constant: float)`
  - `time_constant(n_iterations: int, initial_radius: float) -> float` — applies the `sigma_0 <= 1` clamp and warns.
  - `MIN_RADIUS: float = 1.0 + 1e-9`

**Design note:** the brief derives one time constant from `sigma_0` and uses it
for *both* the radius and the learning rate. That coupling is preserved.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decay.py
from __future__ import annotations

import math

import pytest

from kohonen.decay import (
    MIN_RADIUS,
    ExponentialDecay,
    clamped_radius,
    time_constant,
)


def test_clamped_radius_floors_degenerate_values() -> None:
    assert clamped_radius(0.5) == MIN_RADIUS
    assert clamped_radius(1.0) == MIN_RADIUS


def test_clamped_radius_leaves_normal_values_untouched() -> None:
    assert clamped_radius(5.0) == 5.0


def test_time_constant_matches_the_brief() -> None:
    assert time_constant(100, 5.0) == pytest.approx(100 / math.log(5.0))


@pytest.mark.parametrize("initial_radius", [1.0, 0.5, 0.25])
def test_degenerate_radius_is_clamped_not_raised(initial_radius: float) -> None:
    """A 1x1 or 2x2 grid is legitimate input, not a user error."""
    with pytest.warns(UserWarning, match="initial radius"):
        tau = time_constant(100, initial_radius)
    assert tau > 0
    assert math.isfinite(tau)


def test_clamp_keeps_decay_monotonically_decreasing() -> None:
    with pytest.warns(UserWarning):
        tau = time_constant(50, 1.0)
    decay = ExponentialDecay(MIN_RADIUS, tau)
    values = [decay(t) for t in range(50)]
    assert all(b <= a for a, b in zip(values, values[1:]))


def test_exponential_decay_starts_at_initial_value() -> None:
    assert ExponentialDecay(5.0, 62.0)(0) == pytest.approx(5.0)


def test_exponential_decay_matches_closed_form() -> None:
    decay = ExponentialDecay(0.1, 62.0)
    assert decay(10) == pytest.approx(0.1 * math.exp(-10 / 62.0))


def test_exponential_decay_is_strictly_decreasing() -> None:
    decay = ExponentialDecay(5.0, 62.0)
    values = [decay(t) for t in range(100)]
    assert all(b < a for a, b in zip(values, values[1:]))


def test_time_constant_rejects_non_positive_iterations() -> None:
    with pytest.raises(ValueError, match="n_iterations"):
        time_constant(0, 5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_decay.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'kohonen.decay'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/kohonen/decay.py
"""Decay schedules for the neighbourhood radius and the learning rate."""

from __future__ import annotations

import math
import warnings
from typing import Protocol, runtime_checkable

MIN_RADIUS: float = 1.0 + 1e-9
"""Floor for the initial radius.

The brief derives ``lambda = n_iterations / log(sigma_0)``, which is undefined
at ``sigma_0 == 1`` (a 2x2 grid) and negative below it (a 1x1 grid) -- where a
negative time constant makes the neighbourhood *grow* instead of decay. Small
grids are legitimate input, so the radius is clamped just above 1 rather than
rejected.
"""


@runtime_checkable
class DecaySchedule(Protocol):
    """Maps an iteration number to a decayed value."""

    def __call__(self, t: int) -> float: ...


def time_constant(n_iterations: int, initial_radius: float) -> float:
    """Return ``lambda``, the shared time constant from the brief.

    Warns and clamps when ``initial_radius <= 1``, which would otherwise make
    the time constant undefined or negative.
    """
    if n_iterations <= 0:
        raise ValueError(f"n_iterations must be positive, got {n_iterations}")
    radius = initial_radius
    if radius <= MIN_RADIUS:
        warnings.warn(
            f"initial radius {initial_radius} is too small for a stable time "
            f"constant (grid is 2x2 or smaller); clamping to {MIN_RADIUS}",
            UserWarning,
            stacklevel=2,
        )
        radius = MIN_RADIUS
    return n_iterations / math.log(radius)


def clamped_radius(initial_radius: float) -> float:
    """Return the initial radius, floored at ``MIN_RADIUS``.

    Callers must use this to seed the radius decay, so that a grid small
    enough to trigger the clamp in :func:`time_constant` also decays from the
    clamped value. Clamping one but not the other would leave the two
    inconsistent.
    """
    return max(initial_radius, MIN_RADIUS)


class ExponentialDecay:
    """``value(t) = initial_value * exp(-t / time_constant)``."""

    __slots__ = ("initial_value", "time_constant")

    def __init__(self, initial_value: float, time_constant: float) -> None:
        self.initial_value = initial_value
        self.time_constant = time_constant

    def __call__(self, t: int) -> float:
        return self.initial_value * math.exp(-t / self.time_constant)
```

```python
# src/kohonen/__init__.py -- add the imports
from kohonen.config import TrainingConfig
from kohonen.decay import DecaySchedule, ExponentialDecay

__all__ = ["DecaySchedule", "ExponentialDecay", "TrainingConfig", "__version__"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_decay.py -v --no-cov`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/kohonen/decay.py src/kohonen/__init__.py tests/test_decay.py
git commit -m "Add decay schedules with a guard for degenerate grid sizes"
```

---

### Task 5: Neighbourhood geometry

**Files:**
- Create: `src/kohonen/neighbourhood.py`, `tests/test_neighbourhood.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `coordinate_grid(width: int, height: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]`
  - `squared_distance_to(grid_x, grid_y, bmu_x: int, bmu_y: int) -> NDArray[np.int64]`
  - `gaussian_influence(squared_distance: NDArray[np.floating] | NDArray[np.int64], sigma: float) -> NDArray[np.float64]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neighbourhood.py
from __future__ import annotations

import numpy as np
import pytest

from kohonen.neighbourhood import (
    coordinate_grid,
    gaussian_influence,
    squared_distance_to,
)


def test_coordinate_grid_uses_matrix_indexing() -> None:
    gx, gy = coordinate_grid(3, 2)
    np.testing.assert_array_equal(gx, [[0, 0], [1, 1], [2, 2]])
    np.testing.assert_array_equal(gy, [[0, 1], [0, 1], [0, 1]])


def test_squared_distance_is_zero_at_the_bmu() -> None:
    gx, gy = coordinate_grid(4, 4)
    d2 = squared_distance_to(gx, gy, 2, 1)
    assert d2[2, 1] == 0


def test_squared_distance_matches_pythagoras() -> None:
    gx, gy = coordinate_grid(4, 4)
    d2 = squared_distance_to(gx, gy, 0, 0)
    assert d2[3, 3] == 18  # 3^2 + 3^2


def test_influence_is_one_at_the_bmu() -> None:
    influence = gaussian_influence(np.zeros((3, 3)), sigma=2.0)
    np.testing.assert_allclose(influence, np.ones((3, 3)))


def test_influence_decreases_with_distance() -> None:
    gx, gy = coordinate_grid(5, 5)
    d2 = squared_distance_to(gx, gy, 0, 0)
    influence = gaussian_influence(d2, sigma=2.0)
    assert influence[0, 0] > influence[0, 1] > influence[0, 4]


def test_influence_matches_closed_form() -> None:
    d2 = np.array([[4.0]])
    result = gaussian_influence(d2, sigma=2.0)
    assert result[0, 0] == pytest.approx(np.exp(-4.0 / (2 * 4.0)))


def test_influence_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma"):
        gaussian_influence(np.zeros((2, 2)), sigma=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_neighbourhood.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'kohonen.neighbourhood'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/kohonen/neighbourhood.py
"""Grid geometry and the Gaussian neighbourhood function.

The coordinate grid is built once per training run and reused for every
sample. Rebuilding it inside the sample loop -- or worse, walking it with
nested Python loops -- is what made the original implementation slow.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def coordinate_grid(
    width: int, height: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return per-node x and y coordinates, each shaped ``(width, height)``.

    Uses ``indexing="ij"`` so axis 0 is x and axis 1 is y, matching the
    ``weights[x, y]`` convention of the original implementation.
    """
    return np.meshgrid(  # type: ignore[return-value]
        np.arange(width, dtype=np.int64),
        np.arange(height, dtype=np.int64),
        indexing="ij",
    )


def squared_distance_to(
    grid_x: NDArray[np.int64],
    grid_y: NDArray[np.int64],
    bmu_x: int,
    bmu_y: int,
) -> NDArray[np.int64]:
    """Return the squared grid distance from every node to the BMU.

    Squared distance is returned deliberately: the Gaussian immediately
    squares it again, so taking a square root here would be wasted work.
    """
    dx = grid_x - bmu_x
    dy = grid_y - bmu_y
    return dx * dx + dy * dy


def gaussian_influence(
    squared_distance: NDArray[np.floating] | NDArray[np.int64],
    sigma: float,
) -> NDArray[np.float64]:
    """Return ``exp(-d^2 / (2 * sigma^2))`` for every node."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return np.exp(-squared_distance / (2.0 * sigma * sigma))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_neighbourhood.py -v --no-cov`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/kohonen/neighbourhood.py tests/test_neighbourhood.py
git commit -m "Add vectorised neighbourhood geometry"
```

---

### Task 6: The SOM estimator

**Files:**
- Create: `src/kohonen/som.py`, `tests/test_som.py`
- Modify: `src/kohonen/__init__.py`

**Interfaces:**
- Consumes: `TrainingConfig`, `ExponentialDecay`, `time_constant`, `coordinate_grid`, `squared_distance_to`, `gaussian_influence`
- Produces:
  - `initialise_weights(width: int, height: int, n_features: int, seed: int | None) -> NDArray[np.float64]`
  - `IterationCallback` Protocol: `__call__(iteration: int, weights: NDArray[np.float64], sigma: float, alpha: float) -> None`
  - `SOM(width: int, height: int, *, config: TrainingConfig | None = None)` with `fit(data, *, on_iteration=None) -> SOM`, `predict(data) -> NDArray[np.intp]`, `transform(data) -> NDArray[np.float64]`, `quantisation_error(data) -> float`, attribute `weights_`.

**Design note:** `predict` returns **flat** node indices in `[0, width*height)`.
`np.unravel_index(idx, (width, height))` recovers the grid position. Flat
indices keep the return a plain 1-D integer array, which is what downstream
clustering code expects.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_som.py
from __future__ import annotations

import numpy as np
import pytest

from kohonen import SOM, TrainingConfig
from kohonen.som import initialise_weights


def test_initialise_weights_is_reproducible() -> None:
    a = initialise_weights(4, 3, 5, seed=42)
    b = initialise_weights(4, 3, 5, seed=42)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4, 3, 5)


def test_initialise_weights_differs_across_seeds() -> None:
    a = initialise_weights(4, 3, 5, seed=1)
    b = initialise_weights(4, 3, 5, seed=2)
    assert not np.array_equal(a, b)


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
    som = SOM(4, 4)
    with pytest.raises(AttributeError, match="not fitted"):
        _ = som.weights_


def test_predict_returns_flat_node_indices() -> None:
    data = np.random.default_rng(0).random((6, 3))
    som = SOM(4, 5, config=TrainingConfig(n_iterations=5, seed=0)).fit(data)
    labels = som.predict(data)
    assert labels.shape == (6,)
    assert labels.min() >= 0
    assert labels.max() < 4 * 5


def test_transform_returns_distance_to_every_node() -> None:
    data = np.random.default_rng(0).random((6, 3))
    som = SOM(4, 5, config=TrainingConfig(n_iterations=5, seed=0)).fit(data)
    assert som.transform(data).shape == (6, 20)


def test_predict_agrees_with_transform() -> None:
    data = np.random.default_rng(0).random((6, 3))
    som = SOM(4, 5, config=TrainingConfig(n_iterations=5, seed=0)).fit(data)
    np.testing.assert_array_equal(
        som.predict(data), som.transform(data).argmin(axis=1)
    )


def test_quantisation_error_is_non_negative_and_finite() -> None:
    data = np.random.default_rng(0).random((10, 3))
    som = SOM(5, 5, config=TrainingConfig(n_iterations=20, seed=0)).fit(data)
    error = som.quantisation_error(data)
    assert error >= 0
    assert np.isfinite(error)


def test_training_reduces_quantisation_error() -> None:
    data = np.random.default_rng(0).random((30, 3))
    untrained = SOM(6, 6, config=TrainingConfig(n_iterations=0, seed=0)).fit(data)
    trained = SOM(6, 6, config=TrainingConfig(n_iterations=50, seed=0)).fit(data)
    assert trained.quantisation_error(data) < untrained.quantisation_error(data)


def test_zero_iterations_leaves_weights_at_initialisation() -> None:
    data = np.random.default_rng(0).random((6, 3))
    som = SOM(4, 4, config=TrainingConfig(n_iterations=0, seed=7)).fit(data)
    np.testing.assert_array_equal(som.weights_, initialise_weights(4, 4, 3, 7))


@pytest.mark.parametrize(
    ("width", "height"), [(0, 4), (4, 0), (-1, 4)]
)
def test_invalid_grid_dimensions_are_rejected(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        SOM(width, height)


@pytest.mark.parametrize(
    "bad_data",
    [
        np.array([1.0, 2.0, 3.0]),           # 1-D
        np.zeros((2, 2, 2)),                  # 3-D
        np.empty((0, 3)),                     # no samples
        np.array([[np.nan, 1.0, 2.0]]),       # not finite
        np.array([[np.inf, 1.0, 2.0]]),       # not finite
    ],
)
def test_invalid_training_data_is_rejected(bad_data: np.ndarray) -> None:
    with pytest.raises(ValueError):
        SOM(3, 3, config=TrainingConfig(n_iterations=1, seed=0)).fit(bad_data)


def test_predict_rejects_mismatched_dimensionality() -> None:
    som = SOM(3, 3, config=TrainingConfig(n_iterations=1, seed=0))
    som.fit(np.random.default_rng(0).random((4, 3)))
    with pytest.raises(ValueError, match="features"):
        som.predict(np.random.default_rng(0).random((4, 5)))


def test_fit_does_not_mutate_input_data() -> None:
    data = np.random.default_rng(0).random((6, 3))
    original = data.copy()
    SOM(4, 4, config=TrainingConfig(n_iterations=5, seed=0)).fit(data)
    np.testing.assert_array_equal(data, original)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_som.py -v --no-cov`
Expected: FAIL — `ImportError: cannot import name 'SOM'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/kohonen/som.py
"""The Kohonen Self-Organising Map estimator."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from kohonen.config import TrainingConfig
from kohonen.decay import ExponentialDecay, clamped_radius, time_constant
from kohonen.neighbourhood import (
    coordinate_grid,
    gaussian_influence,
    squared_distance_to,
)


class IterationCallback(Protocol):
    """Observer invoked once per training iteration.

    Implementations must treat ``weights`` as read-only; it is the live
    training array, not a copy. Copy it if you need to retain it.
    """

    def __call__(
        self,
        iteration: int,
        weights: NDArray[np.float64],
        sigma: float,
        alpha: float,
    ) -> None: ...


def initialise_weights(
    width: int, height: int, n_features: int, seed: int | None
) -> NDArray[np.float64]:
    """Draw uniform random weights in ``[0, 1)`` from a seeded generator."""
    rng = np.random.default_rng(seed)
    return rng.random((width, height, n_features))


def _validate_data(data: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"data must be 2-D with shape (n_samples, n_features), "
            f"got {array.ndim}-D with shape {array.shape}"
        )
    if array.shape[0] == 0:
        raise ValueError("data must contain at least one sample")
    if array.shape[1] == 0:
        raise ValueError("data must contain at least one feature")
    if not np.isfinite(array).all():
        raise ValueError("data must contain only finite values (no NaN or inf)")
    return array


class SOM:
    """A Kohonen Self-Organising Map.

    Follows scikit-learn's conventions -- hyperparameters in ``__init__``,
    learned state suffixed with an underscore, ``fit`` returning ``self`` --
    without taking a dependency on scikit-learn.

    Example:
        >>> import numpy as np
        >>> data = np.random.default_rng(0).random((20, 3))
        >>> som = SOM(10, 10, config=TrainingConfig(n_iterations=100, seed=0))
        >>> labels = som.fit(data).predict(data)
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        config: TrainingConfig | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(
                f"width and height must be positive, got {width}x{height}"
            )
        self.width = width
        self.height = height
        self.config = config if config is not None else TrainingConfig()
        self._weights: NDArray[np.float64] | None = None

    @property
    def weights_(self) -> NDArray[np.float64]:
        """The learned node weights, shaped ``(width, height, n_features)``."""
        if self._weights is None:
            raise AttributeError(
                "SOM is not fitted; call fit() before accessing weights_"
            )
        return self._weights

    @property
    def n_nodes(self) -> int:
        """Total node count, ``width * height``."""
        return self.width * self.height

    def fit(
        self,
        data: NDArray[np.floating],
        *,
        on_iteration: IterationCallback | None = None,
    ) -> SOM:
        """Train the map.

        Args:
            data: Training vectors, shaped ``(n_samples, n_features)``.
            on_iteration: Optional observer called once per iteration. When
                ``None`` (the default) there is no overhead.

        Returns:
            ``self``, for chaining.
        """
        array = _validate_data(data)
        n_features = array.shape[1]
        cfg = self.config

        weights = initialise_weights(
            self.width, self.height, n_features, cfg.seed
        )

        if cfg.n_iterations > 0:
            sigma_0 = cfg.resolved_radius(self.width, self.height)
            tau = time_constant(cfg.n_iterations, sigma_0)
            sigma_at = ExponentialDecay(clamped_radius(sigma_0), tau)
            alpha_at = ExponentialDecay(cfg.initial_learning_rate, tau)
            grid_x, grid_y = coordinate_grid(self.width, self.height)
            grid_shape = (self.width, self.height)

            for t in range(cfg.n_iterations):
                sigma_t = sigma_at(t)
                alpha_t = alpha_at(t)
                for sample in array:
                    diff = weights - sample
                    flat_bmu = int(
                        np.argmin(np.einsum("ijk,ijk->ij", diff, diff))
                    )
                    bmu_x, bmu_y = np.unravel_index(flat_bmu, grid_shape)
                    d2 = squared_distance_to(
                        grid_x, grid_y, int(bmu_x), int(bmu_y)
                    )
                    theta = gaussian_influence(d2, sigma_t)
                    weights -= (alpha_t * theta)[..., None] * diff
                if on_iteration is not None:
                    on_iteration(t, weights, sigma_t, alpha_t)

        self._weights = weights
        return self

    def transform(self, data: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return Euclidean distances from each sample to every node.

        Returns:
            Array shaped ``(n_samples, width * height)``.
        """
        array = self._validate_for_inference(data)
        flat = self.weights_.reshape(self.n_nodes, -1)
        diff = array[:, None, :] - flat[None, :, :]
        return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))

    def predict(self, data: NDArray[np.floating]) -> NDArray[np.intp]:
        """Return the flat index of each sample's best matching unit.

        Recover grid coordinates with
        ``np.unravel_index(result, (som.width, som.height))``.
        """
        return np.asarray(self.transform(data).argmin(axis=1), dtype=np.intp)

    def quantisation_error(self, data: NDArray[np.floating]) -> float:
        """Mean distance from each sample to its best matching unit."""
        return float(self.transform(data).min(axis=1).mean())

    def _validate_for_inference(
        self, data: NDArray[np.floating]
    ) -> NDArray[np.float64]:
        array = _validate_data(data)
        expected = self.weights_.shape[2]
        if array.shape[1] != expected:
            raise ValueError(
                f"data has {array.shape[1]} features but the map was fitted "
                f"on {expected} features"
            )
        return array
```

- [ ] **Step 4: Add the degenerate-grid regression test**

This is the division by zero from the original. `clamped_radius` seeds the
radius decay so that a grid small enough to trigger the clamp in
`time_constant` also decays from the clamped value.

```python
# append to tests/test_som.py
@pytest.mark.parametrize(("width", "height"), [(1, 1), (2, 2)])
def test_degenerate_grids_train_without_error(width: int, height: int) -> None:
    """The original divided by zero here."""
    data = np.random.default_rng(0).random((4, 3))
    with pytest.warns(UserWarning, match="initial radius"):
        som = SOM(
            width, height, config=TrainingConfig(n_iterations=10, seed=0)
        ).fit(data)
    assert np.isfinite(som.weights_).all()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_som.py -v --no-cov`
Expected: PASS (all tests)

- [ ] **Step 6: Update the package exports**

```python
# src/kohonen/__init__.py
from __future__ import annotations

from kohonen.config import TrainingConfig
from kohonen.decay import DecaySchedule, ExponentialDecay
from kohonen.som import SOM, IterationCallback

__version__ = "0.1.0"

__all__ = [
    "SOM",
    "DecaySchedule",
    "ExponentialDecay",
    "IterationCallback",
    "TrainingConfig",
    "__version__",
]
```

- [ ] **Step 7: Run all tests and commit**

Run: `uv run --extra dev pytest -v --no-cov && uv run --extra dev mypy && uv run --extra dev ruff check .`
Expected: all pass

```bash
git add src/kohonen tests/test_som.py
git commit -m "Add vectorised SOM estimator with sklearn-style API"
```

---

### Task 7: Prove equivalence with the oracle

**Files:**
- Create: `tests/test_equivalence.py`

**Interfaces:**
- Consumes: `SOM`, `TrainingConfig`, `initialise_weights`, `reference_train`
- Produces: nothing (test-only task)

**This is the task the whole refactor rests on.** It is what converts "I rewrote
it and it looks similar" into "it computes the same thing".

- [ ] **Step 1: Write the failing test**

```python
# tests/test_equivalence.py
"""The vectorised implementation must compute what the original computed."""

from __future__ import annotations

import numpy as np
import pytest

from kohonen import SOM, TrainingConfig
from kohonen._reference import reference_train
from kohonen.som import initialise_weights

TOLERANCE = 1e-12


@pytest.mark.parametrize("width,height", [(4, 4), (5, 3), (8, 8), (3, 7)])
@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("seed", [0, 7, 123])
def test_matches_reference_implementation(
    width: int, height: int, n_features: int, seed: int
) -> None:
    n_iterations = 20
    data = np.random.default_rng(seed).random((6, n_features))
    expected = reference_train(
        data,
        n_iterations,
        width,
        height,
        initialise_weights(width, height, n_features, seed),
    )

    actual = SOM(
        width,
        height,
        config=TrainingConfig(n_iterations=n_iterations, seed=seed),
    ).fit(data).weights_

    np.testing.assert_allclose(actual, expected, atol=TOLERANCE, rtol=0)


def test_matches_reference_on_the_notebooks_own_example() -> None:
    """The 10x10 / 100-iteration configuration from the challenge notebook."""
    data = np.random.default_rng(0).random((10, 3))
    expected = reference_train(
        data, 100, 10, 10, initialise_weights(10, 10, 3, 0)
    )
    actual = SOM(
        10, 10, config=TrainingConfig(n_iterations=100, seed=0)
    ).fit(data).weights_

    np.testing.assert_allclose(actual, expected, atol=TOLERANCE, rtol=0)


def test_deviation_is_within_floating_point_noise() -> None:
    """Report the actual deviation, so drift becomes visible."""
    data = np.random.default_rng(3).random((8, 3))
    expected = reference_train(data, 50, 6, 6, initialise_weights(6, 6, 3, 3))
    actual = SOM(6, 6, config=TrainingConfig(n_iterations=50, seed=3)).fit(
        data
    ).weights_

    deviation = float(np.abs(actual - expected).max())
    assert deviation < TOLERANCE, f"max deviation {deviation:.3e}"
```

- [ ] **Step 2: Run the tests**

Run: `uv run --extra dev pytest tests/test_equivalence.py -v --no-cov`
Expected: PASS (38 tests). If any fail, the vectorised loop has diverged from
the oracle — fix `som.py`, never loosen `TOLERANCE`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_equivalence.py
git commit -m "Prove vectorised trainer matches the original to 1e-12"
```

---

### Task 8: Property-based and regression tests

**Files:**
- Create: `tests/test_properties.py`

**Interfaces:**
- Consumes: `SOM`, `TrainingConfig`
- Produces: nothing (test-only task)

- [ ] **Step 1: Write the tests**

```python
# tests/test_properties.py
"""Invariants that must hold for any valid input, not just chosen examples."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from kohonen import SOM, TrainingConfig

SLOW = settings(max_examples=25, deadline=None)


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
    som = SOM(
        width, height, config=TrainingConfig(n_iterations=5, seed=seed)
    ).fit(data)

    flat = som.weights_.reshape(som.n_nodes, -1)
    for sample, predicted in zip(data, som.predict(data)):
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
    som = SOM(
        width, height, config=TrainingConfig(n_iterations=20, seed=seed)
    ).fit(data)

    lower = min(0.0, float(data.min()))
    upper = max(1.0, float(data.max()))
    assert som.weights_.min() >= lower - 1e-12
    assert som.weights_.max() <= upper + 1e-12


@SLOW
@given(
    n_iterations=st.integers(min_value=2, max_value=60),
    seed=st.integers(min_value=0, max_value=999),
)
def test_sigma_and_alpha_decrease_monotonically(
    n_iterations: int, seed: int
) -> None:
    data = np.random.default_rng(seed).random((4, 3))
    sigmas: list[float] = []
    alphas: list[float] = []

    def record(
        iteration: int, weights: np.ndarray, sigma: float, alpha: float
    ) -> None:
        sigmas.append(sigma)
        alphas.append(alpha)

    SOM(
        5, 5, config=TrainingConfig(n_iterations=n_iterations, seed=seed)
    ).fit(data, on_iteration=record)

    assert all(b < a for a, b in zip(sigmas, sigmas[1:]))
    assert all(b < a for a, b in zip(alphas, alphas[1:]))


GOLDEN = Path(__file__).parent / "fixtures" / "golden_weights_8x8_seed0.npy"


def _golden_weights() -> np.ndarray:
    """Train the configuration the golden fixture pins."""
    data = np.random.default_rng(0).random((10, 3))
    return SOM(8, 8, config=TrainingConfig(n_iterations=50, seed=0)).fit(
        data
    ).weights_


def test_golden_regression_fixture() -> None:
    """Pin numerical output so unintended drift is caught in CI.

    Regenerate deliberately with ``python -m tests.regenerate_golden`` and
    review the diff -- a change here means the algorithm changed.
    """
    np.testing.assert_allclose(
        _golden_weights(), np.load(GOLDEN), atol=1e-12, rtol=0
    )
```

Add `from pathlib import Path` to the imports.

- [ ] **Step 2: Generate and commit the golden fixture**

A fixture file is used rather than a pasted scalar so the whole weight array is
pinned, not a checksum that could collide, and so regeneration is an explicit,
reviewable act.

```python
# tests/regenerate_golden.py
"""Regenerate the golden weights fixture. Run deliberately; review the diff."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kohonen import SOM, TrainingConfig

OUTPUT = Path(__file__).parent / "fixtures" / "golden_weights_8x8_seed0.npy"


def main() -> None:
    data = np.random.default_rng(0).random((10, 3))
    weights = SOM(8, 8, config=TrainingConfig(n_iterations=50, seed=0)).fit(
        data
    ).weights_
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT, weights)
    print(f"Wrote {OUTPUT} with shape {weights.shape}")


if __name__ == "__main__":
    main()
```

Run: `uv run --extra dev python tests/regenerate_golden.py`
Expected: writes `tests/fixtures/golden_weights_8x8_seed0.npy`, shape `(8, 8, 3)`

**Before committing the fixture, confirm Task 7's equivalence tests pass.** The
fixture is only trustworthy if the implementation it captures is already proven
against the oracle; generating it from an unverified implementation would pin
a bug in place.

- [ ] **Step 3: Run tests and check coverage**

Run: `uv run --extra dev pytest -v`
Expected: PASS, and coverage ≥ 95% (the gate is active here).

If coverage falls short, the report names the uncovered lines — add tests for
those specific branches rather than lowering the gate.

- [ ] **Step 4: Commit**

```bash
git add tests/test_properties.py
git commit -m "Add property-based invariants and a golden regression checksum"
```

---

### Task 9: CI pipeline

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the full test suite
- Produces: nothing

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv run --extra dev ruff check .
      - run: uv run --extra dev ruff format --check .
      - run: uv run --extra dev mypy

  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv run --python ${{ matrix.python-version }} --extra dev pytest -v

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
```

- [ ] **Step 2: Verify the same commands pass locally**

Run: `uv run --extra dev ruff check . && uv run --extra dev ruff format --check . && uv run --extra dev mypy && uv run --extra dev pytest -v`
Expected: all pass

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows/ci.yml
git commit -m "Add CI: lint, strict types, and tests across Python 3.10-3.13"
git push -u origin main
```

- [ ] **Step 4: Confirm CI is green**

Run: `gh run watch`
Expected: all jobs pass. Do not proceed to Task 10 until green.

---

### Task 10: Three-way benchmark harness

**Files:**
- Create: `benchmarks/__init__.py`, `benchmarks/compare.py`, `benchmarks/results/.gitkeep`
- Modify: `pyproject.toml` (add `somoclu` to the `bench` extra as optional)

**Interfaces:**
- Consumes: `SOM`, `TrainingConfig`, `reference_train`, `initialise_weights`
- Produces: `benchmarks/results/comparison.json` with schema:
  ```json
  {"generated_at": "...", "platform": {...},
   "runs": [{"implementation": "original|vectorised|minisom|somoclu",
             "width": 10, "height": 10, "n_iterations": 100, "n_samples": 10,
             "n_features": 3, "seconds": 0.31, "weight_updates": 100000,
             "seconds_per_update": 3.1e-06, "quantisation_error": 0.12,
             "notes": "..."}]}
  ```

**Fairness requirement from the spec:** normalise on *weight updates
performed*, not iterations, and state algorithmic differences in the output.
MiniSom uses asymptotic decay `1/(1+t/(T/2))`, not exponential — that goes in
the `notes` field of every MiniSom row.

- [ ] **Step 1: Write the harness**

```python
# benchmarks/compare.py
"""Three-way comparison: original, vectorised, and existing packages.

Writes machine-readable JSON to ``benchmarks/results/comparison.json``. Every
figure quoted in the README, REVIEW, or the published site reads from that
file -- no number is ever transcribed by hand.
"""

from __future__ import annotations

import json
import platform
import sys
import time
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
) -> Run:
    w0 = initialise_weights(width, height, data.shape[1], 0)
    start = time.perf_counter()
    weights = reference_train(data, n_iterations, width, height, w0)
    elapsed = time.perf_counter() - start
    return _make_run(
        "original", elapsed, weights, data, width, height, n_iterations,
        "Triple-nested Python loop, as written in the challenge notebook.",
    )


def bench_vectorised(
    data: NDArray[np.float64], width: int, height: int, n_iterations: int
) -> Run:
    som = SOM(width, height, config=TrainingConfig(n_iterations=n_iterations, seed=0))
    start = time.perf_counter()
    som.fit(data)
    elapsed = time.perf_counter() - start
    return _make_run(
        "vectorised", elapsed, som.weights_, data, width, height, n_iterations,
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
        width, height, data.shape[1], sigma=max(width, height) / 2,
        learning_rate=0.1, random_seed=0,
    )
    start = time.perf_counter()
    som.train(data, n_iterations * data.shape[0], use_epochs=False)
    elapsed = time.perf_counter() - start
    return _make_run(
        "minisom", elapsed, np.asarray(som.get_weights()), data,
        width, height, n_iterations, MINISOM_NOTE,
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
        "somoclu", elapsed, weights, data, width, height, n_iterations,
        "C++/OpenMP, batch SOM. A different algorithm and parallelised; "
        "included to show where an existing library beats this one.",
    )


def main() -> None:
    configurations = [
        (10, 10, 100),
        (30, 30, 100),
        (100, 100, 10),
    ]
    rng = np.random.default_rng(0)
    runs: list[Run] = []

    for width, height, n_iterations in configurations:
        data = rng.random((10, 3))
        print(f"--- {width}x{height}, {n_iterations} iterations ---")
        for fn in (bench_original, bench_vectorised, bench_minisom, bench_somoclu):
            run = fn(data, width, height, n_iterations)
            if run is None:
                print(f"  {fn.__name__:20s} SKIPPED (not installed)")
                continue
            runs.append(run)
            print(f"  {run.implementation:20s} {run.seconds:8.3f}s")

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
```

- [ ] **Step 2: Add somoclu as an optional bench dependency**

In `pyproject.toml`, leave `somoclu` out of the `bench` extra — it fails to
build on Windows and the harness already skips it gracefully. Document this in
the README instead. No change needed to `pyproject.toml`.

- [ ] **Step 3: Run the benchmark**

Run: `uv run --extra dev --extra bench python benchmarks/compare.py`
Expected: prints a table; writes `benchmarks/results/comparison.json`.
The `100x100` original run takes roughly 30 seconds — this is expected.

- [ ] **Step 4: Sanity-check the output**

Run: `uv run python -c "import json,pathlib; d=json.loads(pathlib.Path('benchmarks/results/comparison.json').read_text()); print(len(d['runs']), 'runs'); [print(r['implementation'], r['width'], round(r['seconds'],4)) for r in d['runs']]"`
Expected: at least 6 runs (original + vectorised across 3 configurations).
Confirm the vectorised runs are faster than the original at every size.

- [ ] **Step 5: Commit**

```bash
git add benchmarks pyproject.toml
git commit -m "Add three-way benchmark harness normalised on weight updates"
```

---

### Task 11: Training snapshot exporter

**Files:**
- Create: `benchmarks/export_snapshots.py`

**Interfaces:**
- Consumes: `SOM`, `TrainingConfig`, the `on_iteration` callback
- Produces: `benchmarks/results/snapshots.json` with schema:
  ```json
  {"width": 20, "height": 20, "n_features": 3, "n_iterations": 200,
   "frames": [{"iteration": 0, "sigma": 10.0, "alpha": 0.1,
               "weights": [[...]]}]}
  ```
  `weights` is flattened row-major `(width*height*3)`, rounded to 4 decimals to
  keep the payload small.

- [ ] **Step 1: Write the exporter**

```python
# benchmarks/export_snapshots.py
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
        WIDTH, HEIGHT,
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
                "frames": frames,
            },
            separators=(",", ":"),
        )
    )
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote {OUTPUT} ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
```

The first captured frame is `iteration=0`, which is still close to the random
initialisation and reads correctly as "unorganised" — so the animation opens on
noise without needing a separate pre-training capture.

- [ ] **Step 2: Run the exporter**

Run: `uv run python benchmarks/export_snapshots.py`
Expected: prints frame count and a payload well under 2 MB.

- [ ] **Step 3: Verify the payload size is within budget**

Run: `uv run python -c "import pathlib; p=pathlib.Path('benchmarks/results/snapshots.json'); print(f'{p.stat().st_size/1024/1024:.2f} MB')"`
Expected: under 2 MB. If larger, raise `CAPTURE_EVERY` or lower `DECIMALS`.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/export_snapshots.py benchmarks/results/
git commit -m "Add snapshot exporter driving the explainer animation from real output"
```

---

### Task 12: Written deliverables

**Files:**
- Create: `README.md`, `docs/REVIEW.md`, `docs/adr/0001-productionisation.md`

**Interfaces:**
- Consumes: `benchmarks/results/comparison.json`
- Produces: nothing

**Constraint:** every figure must be read from the JSON and stated with the
platform it was measured on. Do not copy the 157× from the spec — that was a
throwaway probe. Use the committed benchmark's numbers.

- [ ] **Step 1: Read the real numbers**

Run: `uv run python -c "import json,pathlib; d=json.loads(pathlib.Path('benchmarks/results/comparison.json').read_text()); [print(f\"{r['implementation']:12s} {r['width']}x{r['height']} {r['n_iterations']:4d}it {r['seconds']:9.4f}s  qe={r['quantisation_error']:.4f}\") for r in d['runs']]"`

Record the output. Every figure below comes from it.

- [ ] **Step 2: Write `docs/REVIEW.md`**

Structure (prose written at implementation time, figures from Step 1):

1. **Summary** — one paragraph: what the code does, what the review found.
2. **Observations** — the five correctness defects and the structural issues,
   each with the file/line in the notebook and a one-line explanation of the
   consequence. Copy the defect list from the spec's "Correctness defects in
   the original" section.
3. **The five recommendations** — as a table, ordered by business impact, each
   with its supporting evidence. Copy from the spec's "The Five
   Recommendations" section.
4. **Benchmark results** — the table from Step 1, including the MiniSom
   fairness caveat verbatim from `MINISOM_NOTE` and the somoclu result if
   present.
5. **What I did not do, and why** — no GPU backend, no REST API, no CLI.
   State the reasoning. This section is as important as the recommendations.

- [ ] **Step 3: Write `docs/adr/0001-productionisation.md`**

Use the standard ADR format — Status, Context, Decision, Consequences — and
cover the four points from the spec's Productionisation section: batch training
job with MLflow-registered weights, BMU-lookup inference with the explicit
reasoning for *not* building a REST endpoint, quantisation and topographic
error as drift monitors, and the scaling threshold at which a GPU path would
become justified.

- [ ] **Step 4: Write `README.md`**

Sections: what this is; install (`uv sync --extra dev`); 5-line quickstart using
`SOM(...).fit(...).predict(...)`; the headline benchmark table from Step 1 with
the platform noted; how to run tests and benchmarks; a note that `somoclu` is
optional and fails to build on Windows; and a link to `docs/REVIEW.md`.

- [ ] **Step 5: Verify no hand-transcribed figures**

Cross-check every number in the three documents against the JSON from Step 1.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/REVIEW.md docs/adr/
git commit -m "Add review, productionisation ADR, and README with measured figures"
```

---

### Task 13: The published site

**Files:**
- Create: `site/index.html`

**Interfaces:**
- Consumes: `benchmarks/results/comparison.json`, `benchmarks/results/snapshots.json`
- Produces: a published Artifact URL

**Constraints:** strict CSP — all CSS, JS, and data inlined; no CDN, no external
fonts, no runtime fetches. Must be theme-aware (light and dark) and must not
scroll horizontally on mobile.

**REQUIRED SUB-SKILL:** load the `artifact-design` skill before writing the
page, and `artifact-diagramming` if any diagram is added.

- [ ] **Step 1: Build the data payload**

Inline both JSON files into `site/index.html` as `<script type="application/json">`
blocks. Generate the file with a small script rather than pasting by hand, so
regenerating benchmarks refreshes the page:

```bash
uv run python -c "
import json, pathlib
comp = pathlib.Path('benchmarks/results/comparison.json').read_text()
snap = pathlib.Path('benchmarks/results/snapshots.json').read_text()
print(len(comp), len(snap))
"
```

- [ ] **Step 2: Write the three tabs**

- **Review** — observations and the five recommendations, with before/after
  code blocks for the vectorisation change.
- **Benchmarks** — bar charts of `seconds_per_update` by implementation and
  grid size, rendered as inline SVG from the embedded JSON. Include the
  MiniSom caveat as visible text beside the chart, not a footnote.
- **How a SOM Works** — the executive explainer: the core idea before the
  mechanism, a canvas animation with play and scrub controls driven by
  `snapshots.json`, four one-sentence applications, and the honest-limits
  section.

- [ ] **Step 3: Verify locally**

Open `site/index.html` in a browser. Check: tabs switch, the animation plays
from noise to an organised map, charts render, no console errors, no horizontal
scroll at 375px width.

- [ ] **Step 4: Publish**

Use the Artifact tool with `file_path: site/index.html`, a stable `favicon`,
and a one-sentence `description`.

- [ ] **Step 5: Commit**

```bash
git add site/index.html
git commit -m "Add three-tab explainer site driven by real benchmark output"
git push
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Vectorised trainer | 6 |
| Seed injection, dimensionality inference | 3, 6 |
| `sigma_0 <= 1` clamp with warning | 4, 6 |
| Input validation | 6 |
| `TrainingConfig` frozen + hashable | 3 |
| Decay as injectable Protocol | 4 |
| `_reference.py` isolation enforced | 2 |
| Equivalence to 1e-12 | 7 |
| Property-based invariants | 8 |
| Edge cases (1x1, 2x2, zero iterations, NaN) | 6, 8 |
| Golden regression | 8 |
| ≥95% coverage gate | 1, 8 |
| ruff + mypy strict | 1, 9 |
| CI matrix 3.10–3.13 | 9 |
| Three-way benchmark, normalised on updates | 10 |
| somoclu published even if it wins | 10 |
| `on_iteration` callback, zero cost when None | 6 |
| Snapshot exporter | 11 |
| REVIEW.md, ADR, README | 12 |
| Three-tab site | 13 |

No gaps.

**Deferred deliberately:** `radius_cutoff` is validated in `TrainingConfig`
(Task 3) but not yet consumed by `fit`. It is an opt-in approximation and the
spec requires its deviation to be *measured* before it ships. Implementing it
without that measurement would contradict the spec, so it is out of scope for
this plan and tracked as follow-up work.

**Type consistency check:** `initialise_weights(width, height, n_features, seed)`
is used identically in Tasks 6, 7, and 10. `reference_train(input_data,
n_max_iterations, width, height, initial_weights)` is used identically in Tasks
2, 7, and 10. `IterationCallback.__call__(iteration, weights, sigma, alpha)` is
used identically in Tasks 6, 8, and 11. `weights_` is a property everywhere.

**Placeholder scan:** clean. Three defects found in the first draft were fixed
rather than documented — `clamped_radius` now lands in Task 4 where it belongs
(removing a placeholder-then-rework step from Task 6), the snapshot exporter's
dead line is gone, and the tautological golden test was replaced by a committed
`.npy` fixture with an explicit regeneration script.

**Task ordering dependency:** Task 8's golden fixture must be generated only
after Task 7's equivalence tests pass. A fixture generated from an unverified
implementation would pin a bug in place. This is called out in Task 8 Step 2.
