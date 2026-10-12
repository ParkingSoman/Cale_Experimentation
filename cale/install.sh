#!/bin/sh
# Install Cale on a Mac or Linux computer. Run once:  ./cale/install.sh
# This only finds a Python to run cale/install.py. If the computer has none, it gets one from uv.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
for PY in python3 python; do
  if command -v "$PY" >/dev/null 2>&1 && "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
    exec "$PY" "$HERE/install.py" "$@"
  fi
done
echo "==> No Python 3.9 or newer found. Getting one with uv."
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
exec uv run --python 3.12 python "$HERE/install.py" "$@"
