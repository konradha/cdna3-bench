#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PYTHONPATH="$HERE/python:${PYTHONPATH:-}"
export PYTHONPATH
mkdir -p results
SHAPES=(1024,11008,4096 2048,11008,4096 4096,11008,4096 1024,4096,11008 2048,4096,11008)
python -m mxfp4_cdna3.bench --shapes "${SHAPES[@]}" \
  --out results/bench.csv --plot results/bench_violins.png \
  --warmup 50 --iter 100
python -m mxfp4_cdna3.cost_model --shapes "${SHAPES[@]}" \
  --bench results/bench.csv --out results/cost_predictions.csv \
  --plot results/cost_fit.png
