#!/usr/bin/env bash
# Usage: bash scripts/profile.sh <a|b|c|d|e|f|g|h|i|j|k|m> <M> <N> <K> [iters]
# rocprofv3 path. If it aborts on tool/registration init under PyTorch,
# use scripts/profile_v1.sh instead.
set -euo pipefail

STRAT="${1:?strategy: a|b|c|d|e|f|g|h|i|j|k|m}"
M="${2:?M}"
N="${3:?N}"
K="${4:?K}"
ITERS="${5:-10}"

ROCPROF="${ROCPROF:-rocprofv3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/prof_out/${STRAT}_${M}x${N}x${K}"
mkdir -p "$OUT"

PY=(python -m mxfp4_cdna3.profile_one --strategy "$STRAT" --M "$M" --N "$N" --K "$K" --iters "$ITERS")

# --kernel-trace alone is the most permissive of v3's modes under torch.
"$ROCPROF" --kernel-trace --output-format csv -d "$OUT/trace" -- "${PY[@]}"
"$ROCPROF" -i "$ROOT/scripts/rocprof_counters.txt" --output-format csv -d "$OUT/pmc" -- "${PY[@]}"

echo "trace: $OUT/trace/**/*kernel_trace.csv"
echo "pmc:   $OUT/pmc/**/*counter_collection.csv"
