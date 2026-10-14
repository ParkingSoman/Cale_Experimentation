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
    """One way of running (CALE_GPU says which): load the model, read a short prompt, and (unless it is hopelessly slow) a longer piece, answer a few
    tokens, and print the speeds. CALE_PROBE_SKIP_BELOW: stop after the short prompt if it was read slower than this (tokens/s)."""
    import json
    import statistics
    eng = Engine(paths.MODEL)
    eng._tuning = True
    with eng.lock:
        eng._load()
    m = eng.m
    piece = min(256, getattr(m, "batch_tokens", 512))
    eng._run_prompt(48, 8)                                        # warm-up, thrown away

    def speeds(d):
        if d and d.get("tokens", 0) >= 9 and d.get("decode_s", 0) > 0:
            dec = (d["tokens"] - 1) / d["decode_s"]
            return d["prompt_tokens"] / max(d["first_token_s"] - 1.0 / dec, 1e-6), dec
    first = speeds(eng._run_prompt(64, 16))
    out = {"placement": getattr(m, "placement", None)}
    if first:
        piece = int(min(piece, max(64, first[0] * 8)))               # about 8 seconds of reading for a slow reader
    skip = float(os.environ.get("CALE_PROBE_SKIP_BELOW") or 0)
    if first and skip and first[0] < skip:
        out.update(pre=first[0], dec=first[1], first_pre=first[0], dropped=True)
    else:
        runs = [r for r in (speeds(eng._run_prompt(piece, 20)) for _ in range(2)) if r]
        if runs:
            out.update(pre=statistics.median(r[0] for r in runs), dec=statistics.median(r[1] for r in runs), first_pre=first[0] if first else None)
    print("PLACEMENT_PROBE " + (json.dumps(out) if "pre" in out else "failed"), flush=True)
    return 0 if "pre" in out else 1


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
    if os.environ.get("CALE_TUNE_VERBOSE") == "1":
        for e in data["probes"]:
            print(f"  {e['probe']:>14} {e['seconds']:>6} s  {json.dumps(e['result'])[:200]}")
        print("  engine options:", json.dumps(data.get("engine_options"))[:400])
    print(f"Cale is ready for this computer. ({data['seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
