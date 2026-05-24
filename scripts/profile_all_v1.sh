#!/usr/bin/env bash
# Usage: bash scripts/profile_all_v1.sh [M] [N] [K] [iters]
set -uo pipefail
M="${1:-4096}"
N="${2:-11008}"
K="${3:-4096}"
ITERS="${4:-10}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for s in a b c d e f g h i j k m; do
  echo "==== strategy $s ===="
  bash "$ROOT/scripts/profile_v1.sh" "$s" "$M" "$N" "$K" "$ITERS" || echo "[!] strategy $s failed"
done

python -m mxfp4_cdna3.profile_summary --root "$ROOT/prof_out"
