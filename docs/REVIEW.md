# Code Review: Kohonen Self-Organising Map

**Subject:** `notebooks/kohonen.ipynb`

---

## Summary

The algorithm is right and the author clearly understood it — the best-matching-unit
search is already vectorised, so the technique was known. Two lines later, the weight
update is a nested Python loop over every node. That single choice makes the notebook's
own second example take about five minutes instead of about two seconds.

Beneath that sit five correctness defects, one of them an unguarded division by zero.
None would be caught today, because there are no tests.

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

1. **Division by zero on small grids.** `lambda = n_iterations / log(sigma_0)` is
   undefined when `max(width, height) == 2`, because `sigma_0 == 1` and `log(1) == 0`.
   On a 1×1 grid `lambda` is *negative*, so the neighbourhood grows instead of decaying
   and the map actively un-organises.
2. **The radius is computed, then never used as a radius.** The brief selects nodes
   *within* the neighbourhood; the code applies the Gaussian to all nodes
   unconditionally. Defensible mathematically, but undocumented.
3. **`t` is the epoch index, not the iteration counter.** With 10 samples and 100
   epochs, decay advances once per 10 weight updates. The brief is ambiguous here — but
   the choice is silent, so a reader cannot tell whether it was deliberate.
4. **Dimensionality hardcoded to 3.** The brief states a 10-element input implies 10
   weights per node; this accepts RGB and nothing else.
5. **No seed.** `np.random.random` draws from global state, so no run is reproducible.

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
| 2 | **Make it correct and reproducible** — seed injection, guard the degenerate radius, validate inputs, infer dimensionality | Fixes 3 of the 5 defects; see below for why not all 5 |
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

### On recommendation 2 — why only three of five

Three defects are fixed; two are deliberately left in place. The reason is a real
tension, and it is worth stating plainly rather than quietly fixing everything.

**Fixed**, because each is *out-of-band* — none changes what the algorithm computes for
a valid input:

- the division by zero, which only ever fired on grids the original crashed on
- the hardcoded dimensionality, which does not affect the three-feature path
- the missing seed, which changes only the starting point, not the arithmetic

**Not fixed**, because each *does* change the computation:

- the radius is still not used as a cutoff; the Gaussian still applies to every node
- `t` is still the epoch index rather than the iteration counter

Changing either would move the numbers, and the equivalence assertion against
`_reference.py` would fail — correctly. **A rewrite cannot simultaneously prove it
computes what the original computed and change what is computed.** Those are opposite
claims.

So the sequencing matters: establish equivalence first, which makes the vectorisation
provably safe; then change semantics deliberately, one at a time, each with its own
before-and-after evidence. Fixing all five in the same commit as a 155× rewrite would
mean never being able to attribute a behaviour change to its cause.

Both remaining defects are documented above rather than silently carried. They are the
first two items of follow-up work, not oversights.

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
- **The two semantic defects are unfixed by design** — the radius cutoff and the epoch-vs-iteration counter. Fixing either breaks equivalence with the oracle, so they are follow-up work with their own evidence, not part of the rewrite. See "On recommendation 2".
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
