"""Get Cale ready for this computer: measure it once, and keep what was measured. Runs when Cale is installed, and by hand with `cale tune`.

Nothing here is something an app waits for: the installer does it before Cale starts. It takes about a minute (a few seconds on a fast Mac).
Apps, and the people who use them, see none of it.
"""
import os
import sys

from . import paths
from .engine import Engine

STEPS = {"speed": "Timing the model", "pieces": "Trying the graphics card", "threads": "Choosing the best setting"}


def probe():
    """One way of running (CALE_GPU says which): load the model, read a prompt piece and answer a few tokens, and print the speeds."""
    import json
    import statistics
    eng = Engine(paths.MODEL)
    eng._tuning = True
    with eng.lock:
        eng._load()
    m = eng.m
    piece = getattr(m, "batch_tokens", 512)
    eng._run_prompt(64, 12)                                       # warm-up, thrown away
    runs = []
    for _ in range(3):
        d = eng._run_prompt(piece, 24)
        if d and d.get("tokens", 0) >= 9 and d.get("decode_s", 0) > 0:
            dec = (d["tokens"] - 1) / d["decode_s"]
            runs.append((d["prompt_tokens"] / max(d["first_token_s"] - 1.0 / dec, 1e-6), dec))
    print("PLACEMENT_PROBE " + json.dumps({"pre": statistics.median(r[0] for r in runs), "dec": statistics.median(r[1] for r in runs),
                                           "placement": getattr(m, "placement", None)}) if runs else "PLACEMENT_PROBE failed", flush=True)
    return 0 if runs else 1


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="used by `cale tune` itself: time this way of running and print one line")
    ap.add_argument("--profile", default=os.environ.get("CALE_TUNE_PROFILE", "full"), choices=["quick", "full", "max"])
    a = ap.parse_args(argv)
    if a.probe:
        return probe()
    eng = Engine(paths.MODEL)
    data = eng.tune(a.profile, lambda step: print(STEPS.get(step, step), flush=True), report_path=paths.HOME / "tune-report.json")
    print(f"Cale is ready for this computer. ({data['seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
