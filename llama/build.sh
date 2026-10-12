#!/usr/bin/env bash
# Build libcale_llama for Linux or macOS: fetch the pinned llama.cpp, apply our patch, build.
# usage: llama/build.sh [OUT_DIR]      (needs git, cmake, a C++ compiler)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/build}"
TAG="b11558"
SRC="$OUT/llama.cpp"
mkdir -p "$OUT"
if [ ! -d "$SRC" ]; then
  git clone -q --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp.git "$SRC"
  [ -n "${STOCK:-}" ] || git -C "$SRC" apply "$HERE"/patches/*.patch
fi
cmake -S "$HERE" -B "$OUT/b" -DLLAMA_CPP_DIR="$SRC" -DCMAKE_BUILD_TYPE=Release ${STOCK:+-DCALE_STOCK=ON} ${CMAKE_EXTRA:-} >/dev/null
cmake --build "$OUT/b" --config Release -j "$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" --target cale_llama
find "$OUT/b" -name 'libcale_llama.*' -o -name 'cale_llama.dll' | head -3
