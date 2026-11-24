#!/bin/sh
# Run the OCR harness on several prepared datasets, one after another
# (they share one Surya llama.cpp server, so don't run two at once).
#   usage: bench/run_all.sh PYTHON DATASET [DATASET ...]
set -u
PY="$1"; shift
cd "$(dirname "$0")/.."
export DUALOCR_DATA="${DUALOCR_DATA:-$PWD/bench/work}"
for d in "$@"; do
  echo "=== $d $(date +%H:%M:%S)"
  "$PY" bench/run_ocr.py "$d" 2>&1 | grep --line-buffered -vE "HTTP Request|^[[:space:]]*$"
done
echo "=== all done $(date +%H:%M:%S)"
