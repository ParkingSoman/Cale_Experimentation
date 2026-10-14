"""Get Cale ready for this computer: measure it once, and keep what was measured. Runs when Cale is installed, and by hand with `cale tune`.

Nothing here is something an app waits for: the installer does it before Cale starts. It takes about a minute (a few seconds on a fast Mac).
Apps, and the people who use them, see none of it.
"""
import os
import sys

from . import paths
from .engine import Engine

STEPS = {"speed": "Timing the model", "pieces": "Trying the graphics card", "threads": "Choosing the best setting"}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=os.environ.get("CALE_TUNE_PROFILE", "max"), choices=["quick", "full", "max"])
    a = ap.parse_args(argv)
    eng = Engine(paths.MODEL)
    data = eng.tune(a.profile, lambda step: print(STEPS.get(step, step), flush=True), report_path=paths.HOME / "tune-report.json")
    print(f"Cale is ready for this computer. ({data['seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
