# Fixed-Point Benchmark Suite in the Poincare Disk

Reproducible code and precomputed numerical outputs for severe centered and shifted hyperbolic fixed-point benchmarks.

Author: David Ariza Ruiz

## Overview

This repository contains:

- `benchmark_suite.py`: the main simulation script;
- `requirements.txt`: Python dependencies;
- `scripts/run_macos.sh`: a ready-to-run command for macOS;
- `data/`: precomputed run-level outputs and aggregated CSV summaries.

The benchmark suite compares four methods under a matched evaluation budget:

- `GA_0.00`: metric genetic algorithm without refresh;
- `GA_0.10`: metric genetic algorithm with refresh parameter `epsilon = 0.10`;
- `Mann`: Mann iteration;
- `Halpern`: Halpern iteration.

The default configuration is the severe regime used in the experiments:

- contraction parameter `gamma = 0.90`;
- working radius `rho_work = 0.97`;
- population size `N = 100`;
- generations `G = 250`;
- mutation probability `p_m = 0.40`;
- target residual `1e-5`;
- shifted fixed point `z* = (0.55, 0.25)`.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Run the benchmark suite

Default execution:

```bash
python3 benchmark_suite.py --outdir results --runs 500 --jobs 8 --zstar-x 0.55 --zstar-y 0.25
```

Quick smoke test:

```bash
python3 benchmark_suite.py --outdir results_smoke --runs 20 --jobs 4 --zstar-x 0.55 --zstar-y 0.25
```

The script creates:

- `results/data/all_runs.csv`
- `results/data/scenario_manifest.csv`
- `results/data/summary.csv`
- `results/data/paired_comparisons.csv`
- `results/data/manuscript_table.csv`
- `results/data/manuscript_notes.tex`
- `results/figures/*.png`

## Included data

The `data/` directory contains a precomputed release of the severe benchmark experiment with 500 independent runs per scenario. These files can be used directly for manuscript tables, statistical checks, and reproducibility audits without rerunning the full simulation.

## Reproducibility notes

- Random seeds are fixed internally by run index.
- Bootstrap confidence intervals and paired permutation tests are generated reproducibly from a fixed seed.
- The shifted benchmark is implemented intrinsically in the Poincare disk through Mobius operations.
- The Euclidean working ball is a benchmark modelling choice for the computational study.
- Deterministic baselines are matched to the genetic algorithm by the common budget unit "number of evaluations of T".

## Citation and authorship

If you use this code or data, please acknowledge the repository author:

David Ariza Ruiz.
