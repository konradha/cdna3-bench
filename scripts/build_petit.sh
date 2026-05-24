#!/usr/bin/env bash
set -euo pipefail

ROCM="${ROCM:-/opt/rocm-7.2.1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PETIT_DIR="${PETIT_DIR:-$ROOT/third_party/petit-kernel}"

if [[ ! -d "$PETIT_DIR" ]]; then
  mkdir -p "$(dirname "$PETIT_DIR")"
  git clone https://github.com/causalflow-ai/petit-kernel.git "$PETIT_DIR"
fi

if command -v uv &>/dev/null; then
  uv pip install 'cmake>=3.27' ninja tbb-devel
else
  python -m pip install 'cmake>=3.27' ninja tbb-devel
fi

VENV_BIN="$(python -c 'import sys, os; print(os.path.dirname(sys.executable))')"
export PATH="$VENV_BIN:$PATH"
cmake --version | head -1

VENV_ROOT="$(python -c 'import sys; print(sys.prefix)')"
TBB_CMAKE_DIR="$(find "$VENV_ROOT" -name TBBConfig.cmake -printf '%h\n' 2>/dev/null | head -n1)"
if [[ -z "$TBB_CMAKE_DIR" ]]; then
  echo "[!] TBBConfig.cmake not found after installing tbb-devel" >&2
  exit 1
fi
echo "TBB_DIR=$TBB_CMAKE_DIR"

# Wipe stale hunter env vars that Petit's old CMakeLists.txt path imports
unset GFLAGS_ROOT GTEST_ROOT || true

TORCH_DIR="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${ROCM};${TORCH_DIR} -DTBB_DIR=${TBB_CMAKE_DIR}"

export CC="${ROCM}/llvm/bin/clang"
export CXX="${ROCM}/llvm/bin/clang++"
export HIP_CLANG_PATH="${ROCM}/llvm/bin"

rm -rf "$PETIT_DIR/build"

if command -v uv &>/dev/null; then
  ( cd "$PETIT_DIR" && uv pip install --no-build-isolation -v . )
else
  ( cd "$PETIT_DIR" && python -m pip install --no-build-isolation -v . )
fi

python -c "import petit_kernel; print(petit_kernel.__file__)"
