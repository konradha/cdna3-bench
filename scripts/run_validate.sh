#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PYTHONPATH="$HERE/python:${PYTHONPATH:-}"
export PYTHONPATH
python -m mxfp4_cdna3.validate --shapes 256,256,128 512,512,128 1024,1024,256
