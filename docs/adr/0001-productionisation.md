# ADR 0001: Productionising the SOM

**Status:** Accepted
**Date:** 2026-08-24

## Context

The brief asks how this application would be productionised. Answering well
requires first rejecting the question's implied shape.

A Self-Organising Map is **a batch training job that emits an artefact**. It is
not a request-response service. Most "productionise the model" reflexes —
containerise it, put FastAPI in front, autoscale it — are answers to a workload
this is not. Applying them here would produce a system that is harder to
operate and no more useful.

What a SOM actually needs is what any batch ML artefact needs: reproducible
training, a versioned artefact, a way to score new data against it, and a
signal when it has gone stale.

## Decision

### Training: a parameterised batch job

The unit of deployment is a scheduled job, not a running process.

- Orchestrated as an **Azure ML pipeline** or **Databricks job**, whichever the
  surrounding platform already uses. Nothing in the library assumes either.
- `TrainingConfig` is frozen and hashable specifically so a whole run's
  hyperparameters can be logged as **MLflow params** in one call, with no
  hand-maintained parameter dictionary to drift out of sync with the code.
- The trained `weights_` array is registered as an **MLflow model artefact**,
  tagged with the input data version and the git SHA of the training code.
- The `seed` field makes any registered artefact re-derivable. Given the
  config, the data version, and the SHA, the identical map can be reproduced —
  which is what makes an audit trail meaningful rather than decorative.

### Inference: BMU lookup, not an endpoint

Scoring a vector means finding its nearest node — `predict` — which is a
distance computation over `width × height` weights, typically well under a
megabyte.

- For batch scoring, load the artefact and vectorise over the table. This is
  the common case.
- For application use, embed the library directly. The artefact is small enough
  to ship inside the consumer.
- **A REST endpoint is not warranted.** It would add a network hop, a
  deployment surface, and an availability SLO to a computation the caller can
  do locally in microseconds. If a serving layer is ever genuinely needed —
  because the consumer cannot hold the artefact, say — the decision should be
  revisited on that evidence, not assumed up front.

### Monitoring: quantisation and topographic error

A SOM has no labels, so classification metrics do not apply. Two intrinsic
measures serve as drift signals, both tracked per retrain:

- **Quantisation error** — mean distance from each sample to its BMU.
  Implemented as `SOM.quantisation_error`. Rising values mean incoming data no
  longer sits close to the learned map.
- **Topographic error** — the fraction of samples whose best and second-best
  matching units are not adjacent on the grid. Measures whether the map's
  topology still reflects the data's structure. *Not yet implemented; it is the
  first thing to add before this runs in production.*

Alert on trend across retrains rather than on an absolute threshold. The
absolute value depends on the data's scale and dimensionality and carries
little meaning on its own.

### Scaling: not yet, and here is the trigger

At 100×100 the vectorised implementation trains in about two seconds. A GPU or
numba backend is unjustified at this size, and the ADR records the threshold
rather than leaving it to taste:

- Below roughly **1000×1000 nodes**, NumPy broadcasting is sufficient.
- Above that, the per-iteration allocation of `(width, height, n_features)`
  temporaries begins to dominate, and a `numba` or CuPy backend becomes worth
  measuring.
- The library is structured so that this is a swap of the update step in
  `som.py`, not a rewrite — the neighbourhood geometry and decay policies are
  already separate and backend-agnostic.

**Measure before switching.** The 161× already banked came from removing
interpreter overhead, not from specialised hardware, and that lever only needed
to be pulled once.

## Consequences

**Positive**

- The deployment surface is a scheduled job and a registered artefact. There is
  no service to keep alive, patch, or scale.
- Reproducibility is structural rather than procedural: seed, frozen config,
  and data version make any past artefact re-derivable.
- The library stays dependency-light (NumPy only), so it can be embedded in a
  Databricks job, a Lambda, or an air-gapped edge deployment without
  negotiation.

**Negative**

- No real-time scoring path. If a genuine low-latency requirement appears, a
  serving layer must be designed then — this ADR deliberately defers it.
- Topographic error is specified but unimplemented, so drift detection is
  currently one-dimensional.
- Batch retraining means the map is stale between runs. Acceptable for
  segmentation and exploratory analysis; it would not be for a fraud-detection
  path needing minute-level freshness.

**Revisit this ADR if:** a consumer needs sub-second scoring against a map it
cannot hold locally, grids exceed ~1000×1000, or retrain cadence needs to drop
below daily.
