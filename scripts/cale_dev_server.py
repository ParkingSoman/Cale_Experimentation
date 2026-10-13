"""Start a test Cale on port 47899 with its own home. Works on Mac, Windows and Linux.
Does not touch the installed Cale (port 47821).

usage: python scripts/cale_dev_server.py HOME_DIR
Environment: CALE_MODEL (model folder or .gguf file), CALE_BACKEND (mlx | llama), CALE_IDLE_S (default 6), CALE_LLAMA_LIB.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.update(CALE_HOME=sys.argv[1], CALE_PORT="47899", CALE_DEBUG="1", CALE_AUTO_APPROVE=os.environ.get("CALE_DEV_APPROVE", "1"),
                  CALE_IDLE_S=os.environ.get("CALE_IDLE_S", "6"), PYTHONUNBUFFERED="1")
os.environ.setdefault("CALE_MODEL", str(ROOT / "models" / "gemma-3n-e4b-4bit"))
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(line_buffering=True)
runpy.run_module("cale.server", run_name="__main__")      # in this process, so stopping this one stops the server
