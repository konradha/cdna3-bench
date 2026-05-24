#!/usr/bin/env bash
# Usage: bash scripts/dump_isa.sh <a|b|c|d|e|f|g|h|i|j|k|m> [out.s]
# Emits GPU assembly for a single strategy file. Useful greps:
#   v_mfma_           MFMA count (peak ~= n_MFMAs * 16 cycles)
#   ds_read           LDS reads
#   s_waitcnt         wait barriers (more = more stalls)
#   v_accvgpr         AGPR (good: accumulator in AGPR pool)
#   SGPRs/VGPRs/AGPRs occupancy stats in kernel header
#   ScratchSize       VGPR spill bytes (>0 = spilling = bad)
set -euo pipefail
STRAT="${1:?strategy: a|b|c|d|e|f|g|h|i|j|k|m}"
ROCM="${ROCM:-/opt/rocm-7.2.1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
declare -A FILES=(
  [a]="strategy_a_two_pass.cu"
  [b]="strategy_b_lds_dequant.cu"
  [c]="strategy_c_interleaved.cu"
  [d]="strategy_d_8wave.cu"
  [e]="strategy_e_coverage.cu"
  [f]="strategy_f_16wave.cu"
  [g]="strategy_g_prepacked.cu"
  [h]="strategy_h_swscale.cu"
  [i]="strategy_i_swizzle.cu"
  [k]="strategy_k_sched.cu"
  [j]="strategy_j_synthesis.cu"
  [m]="strategy_m_asm.cu"
)
SRC="${FILES[$STRAT]:?unknown strategy}"
OUT="${2:-$ROOT/prof_out/asm_${STRAT}.s}"
mkdir -p "$(dirname "$OUT")"

TORCH_INC="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/include"

"$ROCM/bin/hipcc" -c -S --cuda-device-only -O2 -std=c++17 \
  --offload-arch=gfx942 \
  -I"$ROOT/csrc" -I"$ROCM/include" -I"$TORCH_INC" \
  -DUSE_ROCM=1 -D__HIP_PLATFORM_AMD__=1 \
  "$ROOT/csrc/$SRC" -o "$OUT"

echo "wrote $OUT"
echo "summary:"
grep -c 'v_mfma_'        "$OUT" | xargs -I{} echo "  v_mfma_ count:  {}"
grep -c 'ds_read'        "$OUT" | xargs -I{} echo "  ds_read count:  {}"
grep -c 's_waitcnt'      "$OUT" | xargs -I{} echo "  s_waitcnt count:{}"
grep -c 'v_accvgpr'      "$OUT" | xargs -I{} echo "  v_accvgpr count:{}"
grep -E 'SGPRs|VGPRs|AGPRs|ScratchSize' "$OUT" | head -10 | sed 's/^/  /'
