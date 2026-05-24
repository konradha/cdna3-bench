#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if [[ -z "${FLUX_URI:-}" ]]; then
  echo "ERROR: not inside a flux instance. Get an allocation first, e.g.:" >&2
  echo "  flux alloc -N 1 -t 1h --queue=pbatch" >&2
  exit 1
fi

: "${FLUX_NODES:=1}"
: "${FLUX_GPUS:=1}"
: "${FLUX_TIME:=15m}"
: "${FLUX_LOG_DIR:=results/flux_logs}"
mkdir -p "$FLUX_LOG_DIR"

ARGS=(-N "$FLUX_NODES" -n 1 -g "$FLUX_GPUS" -t "$FLUX_TIME"
      --output="$FLUX_LOG_DIR/validate-{{id}}.out"
      --error="$FLUX_LOG_DIR/validate-{{id}}.err")
if [[ -n "${FLUX_QUEUE:-}" ]]; then ARGS+=(--queue="$FLUX_QUEUE"); fi

flux submit "${ARGS[@]}" -- bash "$HERE/scripts/run_validate.sh"
