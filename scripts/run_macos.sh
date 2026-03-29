#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -r requirements.txt
python3 benchmark_suite.py --outdir results --runs 500 --jobs 8 --zstar-x 0.55 --zstar-y 0.25
