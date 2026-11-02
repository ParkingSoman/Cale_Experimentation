"""Get Cale ready for this computer: measure it once, and keep what was measured. Runs when Cale is installed, and by hand with `cale tune`.

Nothing here is something an app waits for: the installer does it before Cale starts. It takes a few minutes (about two on a fast Mac).
Apps, and the people who use them, see none of it.
"""
import json
import os
import sys

from . import paths
from .engine import Engine

STEPS = {"speed": "Timing the model", "pieces": "Trying the graphics card", "threads": "Choosing the best setting"}


def probe():
    """One way of running (CALE_GPU says which): load the model and read its speeds in FIXED TIME (the readings of cale/tuning.py: windows, not a
    fixed amount of work), then print one line. CALE_PROBE_SKIP_BELOW: stop after the first reading if the prompt was read
    slower than this (tokens/s): a way that is hopeless is not measured further."""
    import json
    import statistics
    import time
    from .tuning import Tuner
    t_in = time.time()
    boot = round(t_in - float(os.environ["CALE_PROBE_T0"]), 1) if os.environ.get("CALE_PROBE_T0") else None   # launch -> this line: Python and imports start-up
    eng = Engine(paths.MODEL)
    eng._tuning = True
    t_a = time.time()
    with eng.lock:
        eng._load()
    t_b = time.time()
    m = eng.m
    piece = min(256, getattr(m, "batch_tokens", 512))
    tuner = Tuner(eng, "quick")
    full = tuner.full_level()
    tuner.timed(full, window_s=1.0, min_tokens=8)                  # one short settling window, thrown away. (The install test settles longer: it is measuring for good. Here every way
                                                                   # of running is a model load, so the readings are few: one answer window, one piece.)
    t_c = time.time()
    first = tuner.timed(full)                                       # (prompt tokens/s of a short prompt, answer tokens/s)
    out = {"placement": getattr(m, "placement", None)}
    if first:
        piece = int(min(piece, max(64, first[0] * 4)))               # FIXED TIME: about 4 seconds of reading, whatever the speed of the reader
    skip = float(os.environ.get("CALE_PROBE_SKIP_BELOW") or 0)
    if first and skip and first[0] < skip:
        out.update(pre=first[0], dec=first[1], first_pre=first[0], dropped=True)
    elif first:
        long_pre = tuner.read_speed(full, piece)
        out.update(pre=long_pre or first[0], dec=first[1], first_pre=first[0])
    out["phases"] = {"boot_s": boot, "engine_init_s": round(t_a - t_in, 1), "load_s": round(t_b - t_a, 1), "warmup_s": round(t_c - t_b, 1), "measure_s": round(time.time() - t_c, 1)}
    print("PLACEMENT_PROBE " + (json.dumps(out) if "pre" in out else "failed"), flush=True)
    return 0 if "pre" in out else 1


def canary():
    """A trial load for the engine: load the model the way CALE_GPU says, read a short prompt, answer a few tokens. Prints CANARY_OK or crashes.
    It runs in its own process so that a graphics driver that fails (or takes the process down) cannot hurt the engine that asked."""
    eng = Engine(paths.MODEL)
    eng._tuning = True
    with eng.lock:
        eng._load()
    eng._run_prompt(48, 4)
    print("CANARY_OK " + str(getattr(eng.m, "placement", None)), flush=True)
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary", action="store_true", help="used by the engine: a trial load on the graphics card, in this process")
    ap.add_argument("--probe", action="store_true", help="used by `cale tune` itself: time this way of running and print one line")
    ap.add_argument("--profile", default=os.environ.get("CALE_TUNE_PROFILE", "full"), choices=["quick", "full", "max"])
    a = ap.parse_args(argv)
    if a.canary:
        return canary()
    if a.probe:
        return probe()
    eng = Engine(paths.MODEL)
    data = eng.tune(a.profile, lambda step: print(STEPS.get(step, step), flush=True), report_path=paths.HOME / "tune-report.json")
    if os.environ.get("CALE_TUNE_VERBOSE") == "1":
        for e in data["probes"]:
            print(f"  {e['probe']:>14} {e['seconds']:>6} s  {json.dumps(e['result'])[:200]}")
        print("  engine options:", json.dumps(data.get("engine_options"))[:400])
    print(f"Cale is ready for this computer. ({data['seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
