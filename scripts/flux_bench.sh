#!/usr/bin/env bash
# Submit the bench as a Flux job. Env overrides:
#   FLUX_NODES (1), FLUX_GPUS (1), FLUX_TIME (60m), FLUX_QUEUE (unset),
#   FLUX_LOG_DIR (results/flux_logs)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if [[ -z "${FLUX_URI:-}" ]]; then
  echo "ERROR: not inside a flux instance. Get an allocation first, e.g.:" >&2
  echo "  flux alloc -N 1 -t 1h --queue=pbatch" >&2
  echo "then re-run this script from inside it (or use 'flux batch')." >&2
  exit 1
fi

: "${FLUX_NODES:=1}"
: "${FLUX_GPUS:=1}"
: "${FLUX_TIME:=60m}"
: "${FLUX_LOG_DIR:=results/flux_logs}"
mkdir -p "$FLUX_LOG_DIR"

ARGS=(-N "$FLUX_NODES" -n 1 -g "$FLUX_GPUS" -t "$FLUX_TIME"
      --output="$FLUX_LOG_DIR/bench-{{id}}.out"
      --error="$FLUX_LOG_DIR/bench-{{id}}.err")
if [[ -n "${FLUX_QUEUE:-}" ]]; then ARGS+=(--queue="$FLUX_QUEUE"); fi

flux submit "${ARGS[@]}" -- bash "$HERE/scripts/run_bench.sh"
