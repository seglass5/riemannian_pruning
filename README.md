# riemannian-pruning

Geometry-informed, capability-targeted pruning of transformer models.

## Overview

This repository explores pruning transformer networks by leveraging Riemannian
geometry — specifically discrete Ricci curvature — to identify structurally
redundant components while preserving capability-critical subnetworks.

### Key ideas

- **Ollivier–Ricci curvature** on the attention graph identifies edges (heads,
  layers) whose removal causes minimal information-geometric distortion.
- **Capability targeting** preserves curvature topology in subnetworks
  correlated with specific task representations.
- Experiments compare geometry-informed pruning against magnitude and gradient
  baselines across language model benchmarks.

## Repository layout

```
src/
  curvature/   Ricci curvature estimation on transformer graphs
  pruning/     Pruning strategies (geometry, magnitude, random)
  eval/        Evaluation harness (perplexity, downstream tasks)
  models/      Model loading and wrapping utilities
experiments/   Runnable experiment scripts
tests/         Unit and integration tests
configs/       YAML experiment configs
notebooks/     Exploratory notebooks
```

## Quick start

```bash
make install          # pip install in editable mode
make test             # run pytest
make lint             # ruff + black checks
make run-experiment   # runs configs/default.yaml
# override config:
make run-experiment CONFIG=configs/my_run.yaml
```

## Requirements

Python ≥ 3.10, PyTorch ≥ 2.1.
