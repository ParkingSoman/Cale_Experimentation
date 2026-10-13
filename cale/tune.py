"""Get Cale ready for this computer: measure it once, and keep what was measured. Runs when Cale is installed, and by hand with `cale tune`.

Nothing here is something an app waits for: the installer does it before Cale starts. It takes about a minute (a few seconds on a fast Mac).
Apps, and the people who use them, see none of it.
"""
import sys

from . import paths
from .engine import Engine

STEPS = {"speed": "Timing the model", "pieces": "Trying the graphics card", "threads": "Choosing the best setting"}


def main():
    eng = Engine(paths.MODEL)
    eng.tune(lambda step: print(STEPS.get(step, step), flush=True))
    print("Cale is ready for this computer.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
