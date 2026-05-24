#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

: "${FLUX_NODES:=1}"
: "${FLUX_GPUS:=1}"
: "${FLUX_TIME:=60m}"
: "${FLUX_LOG_DIR:=results/flux_logs}"
mkdir -p "$FLUX_LOG_DIR"

ARGS=(-N "$FLUX_NODES" -n 1 -g "$FLUX_GPUS" -t "$FLUX_TIME"
      --output="$FLUX_LOG_DIR/bench-{{id}}.out"
      --error="$FLUX_LOG_DIR/bench-{{id}}.err")
if [[ -n "${FLUX_QUEUE:-}" ]]; then ARGS+=(--queue="$FLUX_QUEUE"); fi

if [[ -n "${FLUX_URI:-}" ]]; then
  flux submit "${ARGS[@]}" -- bash "$HERE/scripts/run_bench.sh"
else
  flux batch "${ARGS[@]}" --wrap bash "$HERE/scripts/run_bench.sh"
fi
