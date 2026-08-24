# Kohonen SOM: From Notebook to Production Package — Design

**Date:** 2026-08-18
**Amended:** 2026-08-24 — added three-way comparative benchmarking, the
communication site, and the executive explainer. Budget revised from ~8h to
~12h.
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

- **Time budget: ~12 hours.** Depth in a few places beats thin coverage
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
7. The benchmark suite produces a three-way comparison (original, vectorised,
   MiniSom) normalised on weight updates performed, with algorithmic
   differences stated in the output rather than hidden.
8. A single published web page carries three tabs — Review, Benchmarks, and an
   executive explainer — and every number and animation on it is generated
   from committed code, not hand-authored.

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
class IterationCallback(Protocol):
    """Observer invoked once per training iteration. Must not mutate weights."""
    def __call__(
        self,
        iteration: int,
        weights: NDArray[np.float64],
        sigma: float,
        alpha: float,
    ) -> None: ...

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    n_iterations: int = 100
    initial_learning_rate: float = 0.1
    initial_radius: float | None = None   # default: max(width, height) / 2
    seed: int | None = None
    radius_cutoff: float | None = None    # in sigma units; None = full grid (default)

class SOM:
    def __init__(self, width: int, height: int, *, config: TrainingConfig = ...) -> None: ...
    def fit(
        self,
        data: NDArray[np.floating],
        *,
        on_iteration: IterationCallback | None = None,
    ) -> Self: ...
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
- **Training observability via an optional callback**, not accumulated state.
  `fit` accepts `on_iteration: (iteration, weights, sigma, alpha) -> None`.
  When it is `None` — the default — there is zero overhead and no memory
  growth. The visualisation exporter, MLflow metric logging, and any future
  progress bar are all consumers of this one hook, so none of them require a
  change to the training loop. Accumulating a `history_` array on the estimator
  was rejected: it would make every training run pay the memory cost of a
  feature most runs do not use.

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

## Comparative Benchmarking

`benchmarks/` produces a three-way comparison: **the original implementation**,
**the vectorised implementation**, and **MiniSom** (the de facto community
package; scikit-learn has no SOM).

### The fairness problem

The original and the vectorised implementation are the same algorithm, which
the `2.2e-16` equivalence result proves. MiniSom is not: its default decay is
asymptotic, `1 / (1 + t / (T/2))`, rather than exponential, and it distinguishes
`train_random` from `train_batch`. A raw wall-clock comparison across all three
would therefore be misleading.

Mitigations, all of which appear in the published output rather than only in
the code:

- Normalise on **weight updates performed**, not iterations, so each library is
  charged for the same quantity of work.
- Report **time per weight update** as the headline figure alongside total
  wall-clock.
- State the algorithmic differences in the results table itself.
- Report **quantisation error** next to the timings, so the comparison shows
  solution quality and not merely speed.

### Honest reporting

`somoclu` (C++/OpenMP) is included on a best-effort basis; it is awkward to
install on Windows, so the harness skips it gracefully when unavailable. It may
outperform the NumPy implementation. **If it does, that result is published.**
Documenting where an existing library beats this one — and stating the grid
size at which the trade flips — is a more complete answer to the brief's
question about existing packages than winning the benchmark would be.

Each benchmark writes machine-readable JSON to `benchmarks/results/`, which is
the single source for every figure quoted in the README and on the site. No
number is ever transcribed by hand.

---

## Communication Artefacts

One published page, three tabs, one URL to screenshare:

| Tab | Audience | Content |
|---|---|---|
| **Review** | Technical interviewer | Observations and the five recommendations, with before/after code |
| **Benchmarks** | Technical interviewer | Three-way comparison charts, generated from `benchmarks/results/*.json` |
| **How a SOM Works** | Executive / non-technical | Animated explainer and applications |

### The executive explainer

The audience is assumed to have no machine-learning background. It answers, in
order: what problem does this solve, what is the algorithm doing, and where
would we use it.

- **The core idea before the mechanism.** A SOM presses many-dimensional data
  onto a two-dimensional map while keeping similar things near each other. The
  RGB example carries this without jargon: colours are three numbers, and the
  trained map sorts them into a smooth gradient nobody programmed.
- **Animated training.** A scrubber and a play control run through snapshots of
  a real training run, from random noise to an organised map.
- **Applications**, each one sentence: customer segmentation, anomaly and fraud
  detection, document and text clustering, sensor and equipment health
  monitoring.
- **Honest limits.** Where a SOM is the wrong tool. An explainer that only
  sells is a sales deck, not an engineering artefact.

### How the animation is produced

The visualisation is driven by **real output from the tested package**, never a
JavaScript reimplementation. `benchmarks/export_snapshots.py` trains a SOM
using the `on_iteration` callback, captures weight snapshots at intervals, and
writes compact JSON to `benchmarks/results/`. The page embeds that JSON and
animates it on a canvas in dependency-free JavaScript.

This is a deliberate trade: viewers scrub through precomputed runs rather than
setting arbitrary parameters live. The gain is that the animation is
demonstrably the library's real behaviour, which a browser reimplementation
could never claim.

Snapshot budget: a 20x20x3 grid at 100 frames is roughly 120k floats, which is
comfortably inside the 16MB page limit at reduced precision.

Constraint: the page runs under a strict content-security policy. Everything —
CSS, JavaScript, data — must be inlined. No CDN, no external fonts, no runtime
fetches.

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
| `benchmarks/` | Three-way comparison harness, plus the snapshot exporter |
| `benchmarks/results/*.json` | Machine-readable results; the only source for quoted figures |
| `docs/REVIEW.md` | Observations and the five recommendations |
| `docs/adr/0001-productionisation.md` | Deployment design |
| `README.md` | Install, quickstart, headline benchmark |
| `.github/workflows/ci.yml` | The CI pipeline |
| Published web page | Three tabs: Review, Benchmarks, executive explainer. One URL to screenshare. |
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
| 6 | `on_iteration` callback, its tests, and the snapshot exporter |
| 7-8 | Three-way benchmark harness, JSON results, MiniSom and somoclu |
| 9 | `REVIEW.md`, ADR, README, all figures read from the JSON |
| 10 | Site: Review and Benchmarks tabs, charts from real results |
| 11 | Site: executive explainer and the training animation |
| 12 | Publish, verify both themes and mobile, dry run the walkthrough |

The ordering is a dependency chain, not a preference: the callback (hour 6)
must exist before snapshots can be exported, the benchmarks (hours 7-8) must
produce JSON before any figure can be quoted, and the site (hours 10-11)
consumes both. Nothing on the site is hand-authored, so nothing on the site can
be built early.
