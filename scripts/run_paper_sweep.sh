#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
export PYTHONPATH
mkdir -p results prof_out

SHAPES=(
  # FFN gate/up: M x 11008 x 4096 (Llama2-7B). m sweep covers decode->prefill.
  128,11008,4096  512,11008,4096  2048,11008,4096  4096,11008,4096  8192,11008,4096
  # FFN down: M x 4096 x 11008 (transposed direction).
  128,4096,11008  512,4096,11008  2048,4096,11008  4096,4096,11008  8192,4096,11008
  # Attention proj + 8K-embedding LLMs.
  4096,4096,4096  4096,8192,8192  8192,8192,8192
  # Llama2-70B FFN: 28672 x 8192.
  4096,28672,8192  8192,28672,8192  4096,8192,28672
)

echo "==== sweep over ${#SHAPES[@]} shapes ===="
python -m mxfp4_cdna3.bench --warmup 50 --iter 100 \
  --out results/sweep.csv --plot results/sweep_violins.png \
  --shapes "${SHAPES[@]}"

echo "==== PMC at production shape 4096x11008x4096 (H I K J M) ===="
for s in h i k j m; do
  bash "$ROOT/scripts/profile_v1.sh" "$s" 4096 11008 4096 100 || true
done
python -m mxfp4_cdna3.profile_summary prof_out > results/pmc_summary.txt 2>&1 || true

echo "==== headline tables ===="
python - <<'PY' | tee results/headline.txt
import pandas as pd

df = pd.read_csv("results/sweep.csv")
df["shape"] = df.apply(lambda r: f"{r.M}x{r.N}x{r.K}", axis=1)
df["tflops"] = 2 * df.M * df.N * df.K / df.ns / 1e3

pivot = df.groupby(["shape", "strategy"])["tflops"].median().unstack()
cols = [c for c in ["PETIT", "H", "I", "J", "M", "F", "D", "G"] if c in pivot.columns]
pivot = pivot[cols].round(1)

print("=== Median TFLOPS by shape x strategy ===")
print(pivot.to_string())

if "PETIT" in pivot.columns and "I" in pivot.columns:
    sp = (pivot["I"] / pivot["PETIT"]).round(2)
    table = pd.DataFrame({"I": pivot["I"], "PETIT": pivot["PETIT"], "I/PETIT": sp})
    print("\n=== Strategy I vs Petit speedup ===")
    print(table.to_string())
PY

echo "==== artifacts ===="
echo "  results/sweep.csv           ($(wc -l < results/sweep.csv) rows)"
echo "  results/sweep_violins.png"
echo "  results/pmc_summary.txt"
echo "  results/headline.txt"
