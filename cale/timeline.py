"""One timeline per open or install: a timestamped mark from each part, in one file, so we can see where the time goes.

Every part (the chat starter, the Cale server, the engine, the installer) calls mark("step"). All parts of one run share a run id:
the first part to call mark() makes it and passes it on to its child processes in the CALE_RUN variable.
Marks are numbers and short words only. A mark never waits for the disk and never raises (it uses cale/samples.py, which has the same rules).
The file is `timeline.jsonl` in the Cale folder. `scripts/timeline_report.py` turns it into a table.
"""
import os
import secrets
import time

from . import paths
from .samples import Samples

FILE = "timeline.jsonl"
_s = None


def run_id():
    """The id of this run. It is made once and passed to child processes."""
    if not os.environ.get("CALE_RUN"):
        os.environ["CALE_RUN"] = secrets.token_hex(3)
    return os.environ["CALE_RUN"]


def mark(step, **fields):
    """Record that `step` happened now. `fields` are numbers or short words (for example seconds=1.2)."""
    global _s
    try:
        if _s is None:
            paths.HOME.mkdir(parents=True, exist_ok=True)
            _s = Samples(paths.HOME / FILE)
        return _s.record("mark", run=run_id(), step=step, pid=os.getpid(), mono=round(time.monotonic(), 3), **fields)
    except Exception:
        return None


def flush():
    """Wait until the marks so far are on disk. A short-lived process needs this before it exits."""
    if _s is not None:
        _s.flush()
