#!/bin/sh
# Remove Cale (Mac or Linux). Finds a Python to run cale/uninstall.py.
HERE="$(cd "$(dirname "$0")" && pwd)"
for PY in python3 python; do command -v "$PY" >/dev/null 2>&1 && exec "$PY" "$HERE/uninstall.py" "$@"; done
echo "No Python found." >&2; exit 1
