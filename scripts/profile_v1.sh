#!/usr/bin/env bash
# Usage: bash scripts/profile_v1.sh <a|b|c|d|e|f|g|h|i|j|k|m> <M> <N> <K> [iters]
set -uo pipefail   # no -e: pmc pass may fail on counters the device doesn't expose

STRAT="${1:?strategy: a|b|c|d|e|f|g|h|i|j|k|m}"
M="${2:?M}"
N="${3:?N}"
K="${4:?K}"
ITERS="${5:-10}"

ROCPROF="${ROCPROF_V1:-/opt/rocm-7.2.1/bin/rocprof}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/prof_out/${STRAT}_${M}x${N}x${K}"
rm -rf "$OUT"; mkdir -p "$OUT"

PY=(python -m mxfp4_cdna3.profile_one --strategy "$STRAT" --M "$M" --N "$N" --K "$K" --iters "$ITERS")

echo "[trace] $STRAT $M $N $K"
"$ROCPROF" --hip-trace --stats -o "$OUT/trace.csv" "${PY[@]}" || echo "[!] trace pass exit $?"
echo "[pmc]   $STRAT $M $N $K"
"$ROCPROF" -i "$ROOT/scripts/rocprof_counters.txt" -o "$OUT/pmc.csv" "${PY[@]}" || echo "[!] pmc pass exit $?"

ls -1 "$OUT" || true
