# Kohonen SOM: From Notebook to Production Package — Design

**Date:** 2026-08-18
**Status:** Approved
**Context:** Technical interview deliverable, Lead Machine Learning Engineer, Mantel Group.

## Purpose

Take the single-function Kohonen Self-Organising Map implementation in
`notebooks/kohonen.ipynb` ("Sam's code") and produce a production-ready,
pip-installable Python package with a full test suite, CI, benchmarks, and a
documented productionisation design.

The repository has a second audience beyond the code: it must support a ~15
minute walkthrough of observations and five improvement recommendations during
a one-hour interview. The repository's job is to make those five
recommendations *credible*, not to be exhaustively toured.

## Constraints

- **Time budget: ~8 hours.** Depth in a few places beats thin coverage
  everywhere.
- **Build the library and CI; document the deployment.** No half-built service,
  CLI, or Docker image. The deployment story is an ADR with a diagram.
- Nothing in the original notebook is deleted. It is the evidence base for the
  review, and Sam's function becomes an executable test oracle.
- Toolchain is `uv` (already installed; no system Python on PATH).

## Success Criteria

1. `uv run pytest` passes with >=95% line coverage, enforced by
   `--cov-fail-under=95`.
2. The vectorised trainer matches Sam's original to `atol=1e-12` across a
   parametrised grid of sizes, dimensionalities, and seeds.
3. A measured speed-up is recorded in the README from a reproducible benchmark
   script (probe measured 157x at 100x100; the final figure comes from the
   committed benchmark, not from the probe).
4. `ruff check`, `ruff format --check`, and `mypy --strict` all pass clean.
5. CI runs green on GitHub Actions across Python 3.10-3.13.
6. `docs/REVIEW.md` states the five recommendations, each backed by evidence
   from the repository.

---

## Findings That Drive the Design

Measured on the original code (probe, Python 3.12, NumPy):

| Configuration | Naive runtime | Vectorised | Speed-up | Max abs. deviation |
|---|---|---|---|---|
| 10x10, 100 iterations | 0.31 s | 0.013 s | 24x | 2.22e-16 |
| 100x100, 10 iterations | 3.00 s | 0.019 s | 157x | 2.22e-16 |

Extrapolated, the notebook's own second example (100x100, 1000 iterations)
takes roughly **five minutes**; the vectorised equivalent takes about **two
seconds**. The deviation of 2.22e-16 is one floating-point ULP: the
optimisation is semantics-preserving, not an approximation.

### Correctness defects in the original

1. **`lambda = n_iterations / log(sigma_0)` divides by zero** when
   `max(width, height) == 2`, because `sigma_0 == 1` and `log(1) == 0`. For a
   1x1 grid `lambda` is negative, so the neighbourhood radius *grows* instead
   of decaying. Unguarded.
2. **The radius is computed but never used as a radius.** The brief's step 4
   selects nodes within the neighbourhood radius; the original applies the
   Gaussian influence to every node unconditionally. This is mathematically
   defensible but undocumented, and it is why the update is O(W*H) per sample
   rather than O(sigma_t^2).
3. **`t` is the epoch index, not the iteration counter.** With 10 samples and
   100 epochs, the decay advances once per 10 weight updates. The brief is
   ambiguous here; the choice is silent, so a reader cannot tell whether it was
   deliberate.
4. **Input dimensionality is hardcoded to 3** in
   `np.random.random((width, height, 3))`, contradicting the brief's statement
   that a 10-element input vector implies 10 weights per node.
5. **No seed.** `np.random.random` uses global state, so results are not
   reproducible.

### Structural issues

No package, tests, type hints, docstrings, or input validation. One 20-line
function performs initialisation, scheduling, and training, and is called from
a `__main__` block that also generates data and writes PNGs. Greek identifiers
(sigma and alpha written as literal Greek characters) are unsearchable and
fragile on non-UTF-8 toolchains.

---

## The Five Recommendations

Ordered by business impact rather than technical novelty. The ordering is
itself part of the deliverable.

1. **Vectorise the neighbourhood update.** Replace the nested `for x / for y`
   loop with broadcast operations over a precomputed coordinate grid.
   Evidence: 157x measured, 2.2e-16 deviation.
2. **Make it correct and reproducible.** Inject the seed, guard the
   `sigma_0 <= 1` case, validate inputs, derive dimensionality from the data.
   Fixes five latent defects including a division by zero.

   The guard is specified rather than left to implementation: when
   `sigma_0 <= 1` the time constant `lambda` is undefined or negative, so the
   implementation clamps `sigma_0` to a floor just above 1
   (`max(sigma_0, 1 + 1e-9)`) before deriving `lambda`, and emits a warning
   naming the grid size. Decay then remains monotonically decreasing for every
   valid grid, including 1x1 and 2x2. Raising instead of clamping was rejected:
   a 2x2 grid is legitimate input, not a user error.
3. **Separate the algorithm from its I/O and configuration.** A typed `SOM`
   class exposing `fit`/`predict`, with decay as an injectable policy. No
   plotting or dataset generation inside the trainer.
4. **Test it like it ships.** Keep the original implementation as the test
   oracle; add property-based tests for the algorithm's invariants.
5. **Do not reinvent the wheel, but know when the wheel is wrong.** Benchmark
   against MiniSom and adopt `scipy.spatial.distance.cdist` where it wins.

The distinguishing move is in (4): **`_reference.py` retains Sam's original
implementation**, and CI asserts the fast path matches it. Deleting the old
code is the common choice; keeping it as an executable oracle is how a
refactor of a numerical routine is made safe in production.

---

## Architecture

```
src/kohonen/
  __init__.py        public API surface, __version__
  som.py             SOM: fit / predict / transform
  config.py          TrainingConfig (frozen, self-validating)
  decay.py           DecaySchedule Protocol + Exponential implementation
  neighbourhood.py   Gaussian influence + optional radius cutoff
  _reference.py      original implementation; test oracle only
  py.typed
```

`_reference.py` is never imported by any other module in `src/`. It exists so
that the test suite can import it; a test asserts this isolation.

### Public API

```python
@dataclass(frozen=True, slots=True)
class TrainingConfig:
    n_iterations: int = 100
    initial_learning_rate: float = 0.1
    initial_radius: float | None = None   # default: max(width, height) / 2
    seed: int | None = None
    radius_cutoff: float | None = None    # in sigma units; None = full grid (default)

class SOM:
    def __init__(self, width: int, height: int, *, config: TrainingConfig = ...) -> None: ...
    def fit(self, data: NDArray[np.floating]) -> Self: ...
    def predict(self, data: NDArray[np.floating]) -> NDArray[np.intp]: ...
    def transform(self, data: NDArray[np.floating]) -> NDArray[np.floating]: ...
    def quantisation_error(self, data: NDArray[np.floating]) -> float: ...

    weights_: NDArray[np.float64]   # learned state, set by fit
```

### Design decisions

- **scikit-learn conventions without the scikit-learn dependency.**
  Hyperparameters in `__init__`, learned state suffixed with an underscore,
  `fit` returns `self`. The API is instantly legible to any data practitioner
  and drops into a `Pipeline` later, without pulling a large transitive
  dependency into a small numerical library. Adopting the interface while
  declining the payload is the considered answer to "do not reinvent the
  wheel".
- **Decay as an injectable `Protocol`.** The brief hardcodes exponential
  decay. Treating it as a policy makes linear and inverse-time schedules new
  classes rather than edits to the training loop, and makes each schedule
  unit-testable in isolation.
- **`n_features` inferred at `fit` time**, removing the hardcoded 3.
- **`TrainingConfig` is frozen and hashable**, so it can be logged wholesale as
  MLflow parameters.

### Numerical approach

Precompute the node coordinate grid once via `np.meshgrid(..., indexing="ij")`.
Per sample: compute `diff = weights - v`, find the BMU with
`argmin(einsum('ijk,ijk->ij', diff, diff))` (no square root is needed for an
argmin), compute squared grid distances against the BMU, form the influence
`theta`, and apply `weights -= (alpha_t * theta)[..., None] * diff`.

Online (per-sample) update semantics are preserved, which is what makes
equivalence with the reference implementation possible.

The `radius_cutoff` option restricts each update to a window of
`radius_cutoff * sigma_t` nodes around the BMU. This is an approximation: nodes
outside the window receive no update at all, where the exact algorithm would
apply a small non-zero influence. It is therefore **off by default**
(`None`), so that the default configuration is exactly equivalent to the
reference implementation. When enabled, it is tested against a measured
tolerance rather than for exact equivalence; the benchmark records the actual
deviation and the speed-up it buys at each grid size, and the README reports
both. No tolerance figure is asserted in advance.

### Explicit non-goal

**No GPU or numba backend.** At 100x100 the vectorised implementation completes
in about two seconds. A CUDA path would be complexity the data does not
justify.

---

## Testing Strategy

Test-driven: each test layer is written before the implementation it pins down.

| Layer | Scope |
|---|---|
| Equivalence | Fast path vs `_reference.py`, parametrised over grid sizes, dimensionalities, and seeds, at `atol=1e-12`. |
| Property-based (`hypothesis`) | The returned BMU is the true argmin; sigma and alpha decrease monotonically; weights remain within the data's per-feature bounds; identical seeds produce identical weights. |
| Edge cases | 1x1 and 2x2 grids (the division by zero), single sample, `n_iterations=0`, NaN and inf inputs, mismatched dimensionality. Each maps to a defect listed above. |
| Regression | Golden hash of the weights for a fixed seed, catching unintended numerical drift. |
| Benchmark | `pytest-benchmark`, reported but not gated, including the MiniSom comparison. |

Coverage gate: `--cov-fail-under=95`.

---

## Tooling and CI

`pyproject.toml` with `uv` for dependency resolution and locking. GitHub
Actions runs, on every push and pull request:

1. `ruff check` and `ruff format --check`
2. `mypy --strict`
3. `pytest` with coverage, matrixed across Python 3.10, 3.11, 3.12, 3.13
4. Wheel build

Benchmarks run in CI for visibility but never fail the build, since runner
performance is too variable to gate on.

---

## Productionisation (ADR)

`docs/adr/0001-productionisation.md` records the deployment design rather than
implementing it. A SOM is a **batch training job**, not an online-inference
product, and the ADR is explicit about that framing:

- **Training**: a parameterised batch job (Azure ML pipeline or Databricks
  job). `TrainingConfig` is logged as MLflow parameters; the trained weights
  are registered as an MLflow model artefact alongside the data version.
- **Inference**: BMU lookup, either as batch scoring over a table or as the
  library embedded in a consuming application. A REST endpoint is not
  warranted, and the ADR says why.
- **Monitoring**: quantisation error and topographic error tracked across
  retrains as drift indicators.
- **Scaling**: the honest position that a GPU path is unnecessary below roughly
  1000x1000 grids, with the threshold at which that changes.

---

## Deliverables

| Artefact | Purpose |
|---|---|
| `src/kohonen/` | The package |
| `tests/` | Test suite, >=95% coverage |
| `benchmarks/` | Reproducible speed-up and MiniSom comparison |
| `docs/REVIEW.md` | Observations and the five recommendations |
| `docs/adr/0001-productionisation.md` | Deployment design |
| `README.md` | Install, quickstart, headline benchmark |
| `.github/workflows/ci.yml` | The CI pipeline |
| Published web page | Screenshare artefact for the walkthrough, generated from `REVIEW.md` |
| `notebooks/kohonen.ipynb` | Unchanged original |

**Commit history is part of the deliverable.** The work lands as roughly ten
atomic, well-messaged commits rather than a single dump.

## Schedule

| Hour | Output |
|---|---|
| 1 | Scaffold, `pyproject.toml`, uv lock, CI skeleton, first commit |
| 2-3 | Core package, test-driven |
| 4 | `_reference.py` and the equivalence suite |
| 5 | Property and edge-case tests to 95% coverage |
| 6 | Benchmarks and MiniSom comparison |
| 7 | `REVIEW.md`, ADR, README |
| 8 | Published web page and a dry run of the walkthrough |
