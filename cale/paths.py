"""Where Cale keeps its files, and the address it listens on."""
import os
from pathlib import Path

from . import oslayer

HOST = "127.0.0.1"
PORT = int(os.environ.get("CALE_PORT", "47821"))
HOME = oslayer.home_dir()
MODEL = Path(os.environ.get("CALE_MODEL", HOME / "models" / ("gemma-3n-e4b" if oslayer.MAC else "gemma-3n-e4b.gguf")))   # Mac: a folder (MLX). Windows, Linux: one .gguf file
KEYS = HOME / "keys.json"
SITES = HOME / "web_sites.json"
LOGS = HOME / "logs"
CURVE = Path(os.environ.get("CALE_CURVE", HOME / "curve.json"))
LABEL = "ai.cale.runtime"
