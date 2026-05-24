#!/usr/bin/env bash
set -uo pipefail

M="${1:-4096}"
N="${2:-11008}"
K="${3:-4096}"
WARMUP="${4:-50}"
ITERS="${5:-100}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
export PYTHONPATH
mkdir -p results prof_out

SHAPE="${M},${N},${K}"

python -m mxfp4_cdna3.validate --shapes "$SHAPE" || true

python -m mxfp4_cdna3.bench --shapes "$SHAPE" \
  --out results/bench_compare.csv --plot results/bench_compare.png \
  --warmup "$WARMUP" --iter "$ITERS"

for s in h i k j m; do
  bash "$ROOT/scripts/profile_v1.sh" "$s" "$M" "$N" "$K" "$ITERS" || true
done

python -m mxfp4_cdna3.profile_summary prof_out || true
