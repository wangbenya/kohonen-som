# Code Review: Kohonen Self-Organising Map

**Subject:** `notebooks/kohonen.ipynb`

---

## Summary

The algorithm is right and the author clearly understood it — the best-matching-unit
search is already vectorised, so the technique was known. Two lines later, the weight
update is a nested Python loop over every node. That single choice makes the notebook's
own second example take about five minutes instead of about two seconds.

Beneath that sit two genuine bugs -- one of them an unguarded division by zero -- a
reproducibility defect, and two places where the brief contradicts itself and the code
picks a side without saying so. None would be caught today, because there are no tests.

Every figure below comes from `benchmarks/results/comparison.json`, produced by
`benchmarks/compare.py`.

---

## Observations

### Efficiency

The BMU search is vectorised:

```python
bmu = np.argmin(np.sum((weights - vt) ** 2, axis=2))
```

The update immediately below it is not:

```python
for x in range(width):
    for y in range(height):
        di = np.sqrt(((x - bmu_x) ** 2) + ((y - bmu_y) ** 2))
        theta_t = np.exp(-(di ** 2) / (2 * (sigma_t ** 2)))
        weights[x, y] += alpha_t * theta_t * (vt - weights[x, y])
```

For the 100×100 example this runs roughly 100 million Python-level iterations, each
allocating NumPy temporaries to update a three-element vector.

The measured cost is the tell:

| Grid | Original, per weight update |
|---|---|
| 10×10 | 3082 ns |
| 30×30 | 3055 ns |
| 100×100 | 3034 ns |

**The cost per node is flat across a hundredfold change in grid size.** That is not
algorithmic cost — it is per-node Python interpreter overhead, invariant to the work
being done. The fix is not a better algorithm; it is removing the interpreter from the
inner loop.

### Correctness

Two of these are unambiguous bugs. One is a reproducibility defect. The last two are
not bugs at all -- they are silent resolutions of a brief that contradicts itself, and
the failing is the silence rather than the choice.

**Bugs**

1. **Division by zero on small grids.** `lambda = n_iterations / log(sigma_0)` is
   undefined when `max(width, height) == 2`, because `sigma_0 == 1` and `log(1) == 0`.
   On a 1x1 grid `lambda` is *negative*, so the neighbourhood grows instead of decaying
   and the map actively un-organises.
2. **Dimensionality hardcoded to 3.** The brief states plainly that a 10-element input
   implies 10 weights per node. `np.random.random((width, height, 3))` accepts RGB and
   nothing else, which directly contradicts it.

**Reproducibility**

3. **No seed.** `np.random.random` draws from global state, so no run is reproducible
   and no result can be audited or compared against another.

**Undocumented interpretation, not error**

4. **The radius is never used as a cutoff.** The brief's prose (step 4) says nodes
   *within* the radius form the neighbourhood, but the brief's own influence formula,
   `exp(-d^2 / 2*sigma_t^2)`, has no cutoff in it. The code follows the formula, which
   is the standard Kohonen formulation -- a Gaussian that decays smoothly over every
   node. Imposing a hard cutoff would be an *approximation for speed*, not a
   correction. The code is defensible here; it simply never says which reading it took.
5. **`t` is the epoch index, not a per-sample counter.** The brief says "enumerate
   through the training data for some number of iterations" and sets
   `n_max_iterations = 100`, which the epoch reading matches: 100 passes over 10
   samples is 1000 updates. The per-sample reading would make the same constant mean 10
   passes, fitting the wording worse. Again defensible, again unstated.

Items 4 and 5 are why this review does not simply "fix all five". Changing either would
substitute one reading of an ambiguous brief for another while claiming to preserve
behaviour -- see [On recommendation 2](#on-recommendation-2--what-was-fixed-and-what-was-preserved).

### Structure

No package, tests, type hints, or input validation. One 20-line function does
initialisation, scheduling, and training; the `__main__` block also generates data and
writes PNGs, so the algorithm cannot be exercised without doing file I/O. Greek
identifiers (`σ0`, `αt`) are unsearchable and fragile on non-UTF-8 toolchains.

---

## Five Recommendations

Ordered by business impact rather than technical novelty.

| # | Recommendation | Evidence |
|---|---|---|
| 1 | **Vectorise the neighbourhood update** — broadcast over a coordinate grid built once per run | 155× at 100×100; deviation 2.220e-16 |
| 2 | **Make it correct and reproducible** — seed injection, guard the degenerate radius, validate inputs, infer dimensionality | Fixes both genuine bugs plus reproducibility; two ambiguities documented, not silently changed |
| 3 | **Separate the algorithm from its I/O** — a typed `SOM` with `fit`/`predict`; no plotting inside the trainer | Enables testing, reuse, non-RGB data |
| 4 | **Test it like it ships** — keep the original as an executable oracle | 52 tests, 100% coverage, CI on 3.10–3.14 |
| 5 | **Don't reinvent the wheel — but know when the wheel is wrong** — benchmark against MiniSom, adopt scikit-learn's conventions without the dependency | See below |

### On recommendation 1

The speed-up is not a fixed number. It **scales with grid size**, because what is being
removed is per-node interpreter overhead:

| Grid | Original | Vectorised | Speed-up |
|---|---|---|---|
| 10×10, 100 iterations | 0.308 s | 0.0130 s | **24×** |
| 30×30, 100 iterations | 2.750 s | 0.0299 s | **92×** |
| 100×100, 10 iterations | 3.034 s | 0.0196 s | **155×** |

### On recommendation 2 — what was fixed, and what was preserved

**Every genuine defect is fixed**: the division by zero, the hardcoded dimensionality,
and the missing seed. None of these three changes what the algorithm computes for valid
input, so the equivalence proof in recommendation 4 still holds: the guard only fires on
grids the original crashed on, and dimensionality inference is exercised at
`n_features=10` in the equivalence suite and matches to 1e-12.

**The two interpretation choices are preserved, and documented.** The absent radius
cutoff and the epoch-scoped `t` are not errors — both follow the brief's own formulas,
as set out under Correctness above. Substituting a different reading would be a
behaviour change dressed up as a bug fix.

There is also a sequencing reason. Changing either would move the numbers, so the
assertion against `_reference.py` would fail — correctly. A rewrite cannot at once prove
it computes what the original computed and change what is computed. Establishing
equivalence first is what makes the 155× vectorisation provably safe; a semantic change
afterwards is a separate, deliberate step carrying its own before-and-after evidence.

If the brief's prose is authoritative rather than its formulas, both are small changes —
a mask on `theta` for the cutoff, and moving the decay inside the sample loop for `t`.
The work is not the edit. It is building a second oracle so the effect on convergence
can be shown rather than asserted.

### On recommendation 3

The original exposes one function that returns a bare array, and **no inference path at
all**. A consumer wanting to map a new vector onto the trained grid has to reimplement
the best-matching-unit search themselves — which is both wasted work and a chance to
implement it differently from the trainer.

The `__main__` block compounds it by generating the dataset, training, and writing PNGs
in the same place, so the algorithm cannot be exercised without doing file I/O. That is
the reason there are no tests: the code is not shaped to be called.

The rewrite separates the three concerns:

| Concern | Where it lives now |
|---|---|
| Hyperparameters | `TrainingConfig` — frozen, validated at construction |
| The algorithm | `SOM.fit` — no I/O, no plotting, no dataset generation |
| Inference | `SOM.predict` / `transform` / `quantisation_error` |

Plotting and data generation move out of the package entirely; they belong to the
caller. The payoff is not tidiness — it is that every piece becomes callable in
isolation, which is what makes the test suite in recommendation 4 possible at all.

### On recommendation 4

`src/kohonen/_reference.py` keeps the original implementation, and CI asserts the fast
path matches it to `atol=1e-12` across 14 cases — three grid shapes, two
dimensionalities, two seeds, plus the notebook's own configuration.

Worst observed deviation: **2.220e-16**, exactly `np.finfo(np.float64).eps`. The
optimisation is *semantics-preserving*, not an approximation.

Most refactors delete the code they replace. Keeping it as an executable oracle is what
makes rewriting a numerical routine safe rather than merely plausible.

### On recommendation 5

scikit-learn has no SOM, so MiniSom is the real comparator.

| Grid | Original | This library | MiniSom |
|---|---|---|---|
| 10×10 | 0.308 s | **0.0130 s** | 0.0176 s |
| 30×30 | 2.750 s | **0.0299 s** | 0.0417 s |
| 100×100 | 3.034 s | **0.0196 s** | 0.0295 s |

**The caveat matters more than the win.** MiniSom is not running the same algorithm — it
uses asymptotic decay `1/(1+t/(T/2))` rather than the exponential decay this brief
specifies, so raw wall-clock is not like-for-like. The benchmark normalises on *weight
updates performed* and records the difference in every MiniSom row of the output JSON.

Quantisation error says the same from the other side: identical between the original and
this library (0.1031 vs 0.1031, as the equivalence proof requires) but different for
MiniSom (0.1148). They are solving slightly different problems.

Where the wheel wins is the *interface*, not the code: hyperparameters in `__init__`,
learned state as `weights_`, `fit` returning `self`. This library follows all three
without taking the dependency — a 200-line numerical package should not pull in a large
transitive tree to borrow a convention.

---

## Productionising This

A SOM is **a batch training job that emits an artefact**, not a request-response service.
Most "productionise the model" reflexes — containerise it, put an API in front, autoscale
— answer a workload this is not.

**Training.** A parameterised batch job (Azure ML pipeline or Databricks job).
`TrainingConfig` is frozen and hashable specifically so a run's hyperparameters log as
MLflow params in one call. The trained `weights_` registers as an MLflow artefact tagged
with the data version and git SHA. With the config, the data version, and the SHA, any
past artefact is re-derivable — which is what makes an audit trail meaningful.

**Inference.** BMU lookup over a weights array typically well under a megabyte. Batch
scoring loads the artefact and vectorises over the table; applications embed the library
directly. A REST endpoint would add a network hop, a deployment surface, and an
availability SLO to a computation the caller can do locally in microseconds.

**Monitoring.** No labels, so classification metrics do not apply. Quantisation error
(implemented) and topographic error (*not yet implemented* — the first thing to add
before this runs in production) both work as drift signals. Alert on trend across
retrains, not an absolute threshold; the absolute value depends on the data's scale.

**Scaling.** Below roughly 1000×1000 nodes, NumPy broadcasting is sufficient. Above
that, per-iteration allocation of `(width, height, n_features)` temporaries begins to
dominate and a numba or CuPy backend becomes worth measuring. Measure before switching —
the 155× already banked came from removing interpreter overhead, not from special
hardware.

---

## What I Did Not Do, And Why

- **No GPU or numba backend.** Two seconds at 100×100 does not justify the complexity.
- **No REST API.** See above — it answers the wrong workload.
- **No CLI or Docker image.** Both straightforward; neither was the bottleneck. The time
  went into the equivalence proof, which is what makes every other claim here
  trustworthy.
- **I did not "fix" the two ambiguities.** The absent radius cutoff and the
  epoch-scoped `t` both follow the brief's own formulas, so changing them would be a
  behaviour change dressed as a bug fix — and would break equivalence with the oracle.
  They are documented instead, and sequenced as separate work with their own evidence.
- **`somoclu` was not benchmarked.** The C++/OpenMP implementation would plausibly beat
  this one at large grids, and that result would have been published had it run — but it
  does not build on Windows. This is a gap in the comparison, not a favourable omission.

---

## Reproducing

```bash
uv run --extra dev pytest                                   # 52 tests, 100% coverage
uv run --extra dev --extra bench python benchmarks/compare.py
```

Figures above: Python 3.13.14, NumPy 2.5.2, Windows, AMD64.
