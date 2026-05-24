#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if [[ -z "${ROCM_PATH:-}" || ! -d "$ROCM_PATH" ]]; then
  if command -v hipcc &>/dev/null; then
    ROCM_PATH="$(dirname "$(dirname "$(readlink -f "$(which hipcc)")")")"
  else
    echo "ERROR: hipcc not found and ROCM_PATH unset. Did you 'module load rocm/7.2'?" >&2
    exit 1
  fi
fi
export ROCM_PATH
: "${PYTORCH_ROCM_ARCH:=gfx942}"
export PYTORCH_ROCM_ARCH

python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("ERROR: torch not installed in the active interpreter. "
             "Are you inside the venv (`source .venv/bin/activate`)? "
             "Install with: uv pip install --extra-index-url "
             "https://download.pytorch.org/whl/rocm7.2 "
             "--index-strategy unsafe-best-match torch")
if torch.version.hip is None:
    sys.exit(f"ERROR: torch {torch.__version__} is not a ROCm build (torch.version.hip is None).")
print(f"using {sys.executable}")
print(f"torch {torch.__version__}, ROCm {torch.version.hip}")
PY

echo "ROCM_PATH=$ROCM_PATH  PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"

# Use uv if available (respects VIRTUAL_ENV); else fall back to `python -m pip`
# which always uses the active python interpreter, not whatever `pip` resolves to on PATH.
if command -v uv &>/dev/null; then
  uv pip install --no-build-isolation -e . -v
else
  python -m pip install --no-build-isolation -e . -v
fi

python -c "from mxfp4_cdna3 import device_info; import json; print(json.dumps(device_info(0), indent=2))"
