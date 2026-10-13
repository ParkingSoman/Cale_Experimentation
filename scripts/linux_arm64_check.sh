#!/bin/sh
# Build Cale's llama.cpp engine for Linux arm64 and run the suites on it, in an Ubuntu arm64 container (docker or colima on an Apple silicon Mac).
# usage: scripts/linux_arm64_check.sh MODEL.cale.gguf [suite,suite,...]
# The container reads this repo and the model through mounts. Nothing is installed on the computer.
set -e
MODEL="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
ONLY="${2:-oslayer,choose,constrain,backend_llama,honest,objects,goals,structured,promptcache,crowd,anylang}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
docker run --rm --platform linux/arm64 -v "$ROOT":/work:ro -v "$(dirname "$MODEL")":/models:ro -v cale-arm64-build:/build \
  -e ONLY="$ONLY" -e MODEL_FILE="/models/$(basename "$MODEL")" ubuntu:24.04 bash -c '
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq build-essential gcc-14 g++-14 cmake git python3 python3-pip ninja-build curl >/dev/null   # gcc 14: the newest ARM variants (SME) need it
export CC=gcc-14 CXX=g++-14
uname -m; nproc; free -g | head -2
mkdir -p /code && cp -r /work/cale /work/nest /work/scripts /work/llama /work/docs /code/ 2>/dev/null; cd /code
rm -rf /build/b                                                  # a new compiler needs a new build folder
CALE_BACKENDS_DL=1 bash llama/build.sh /build 2>&1 | grep -E "error|Error|Built target cale_llama|libcale_llama" | tail -6
cp /build/b/libcale_llama.so /build/b/bin/* /tmp/ 2>/dev/null || true
mkdir -p /lib_out && cp /build/b/libcale_llama.so /lib_out/ && cp -a /build/b/bin/*.so* /lib_out/
python3 -c "import sys; print(sys.version)"
ldd /lib_out/libcale_llama.so | sed "s/(0x[0-9a-f]*)//" | head -8
python3 scripts/run_suites.py "$MODEL_FILE" --lib /lib_out/libcale_llama.so --only "$ONLY" 2>&1 | sed -n "/== summary/,\$p"
'
