#!/bin/sh
# Build and run llama/tests/test_unrepack.cpp against the static ggml libraries of a finished Mac or Linux build (llama/build.sh OUT).
# usage: scripts/check_unrepack.sh OUT_DIR      (the folder you gave to llama/build.sh, without CALE_BACKENDS_DL)
set -e
OUT="$(cd "$1" && pwd)"; ROOT="$(cd "$(dirname "$0")/.." && pwd)"
B="$OUT/b/llama.cpp/ggml/src"; S="$OUT/llama.cpp"
EXTRA=""; [ "$(uname)" = "Darwin" ] && EXTRA="-framework Accelerate -framework Foundation"
${CXX:-c++} -std=c++17 ${CXXFLAGS:-} -I"$S/ggml/include" -I"$S/ggml/src" -I"$S/ggml/src/ggml-cpu" "$ROOT/llama/tests/test_unrepack.cpp" \
  "$B/libggml.a" "$B/libggml-cpu.a" "$B/libggml-base.a" $EXTRA -lpthread -o "$OUT/test_unrepack"
"$OUT/test_unrepack"
