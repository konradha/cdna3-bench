#!/usr/bin/env bash
# Usage: bash scripts/profile_all.sh [M] [N] [K] [iters]
# rocprofv3 path. Falls back to scripts/profile_all_v1.sh if v3 trips on torch's late dlopen.
set -euo pipefail
M="${1:-4096}"
N="${2:-11008}"
K="${3:-4096}"
ITERS="${4:-10}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for s in a b c d e f g h i j k m; do
  echo "==== strategy $s ===="
  bash "$ROOT/scripts/profile.sh" "$s" "$M" "$N" "$K" "$ITERS"
done

python -m mxfp4_cdna3.profile_summary --root "$ROOT/prof_out"
