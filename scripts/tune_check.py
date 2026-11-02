"""Run the real hardware test (`cale tune`) a few times and say how long it takes, how stable it is, and how close it is to a slow, careful measurement.

usage: python scripts/tune_check.py WORK_FOLDER --truth results/sizes-accuracy-NAME.json [--runs 3] [--profile full]
  WORK_FOLDER  a new folder. Each run gets its own Cale home inside it (nothing real is touched).
  --truth      the file made by scripts/sizes_accuracy.py on THIS computer: every size measured three times with long readings.
Uses CALE_MODEL / CALE_BACKEND / CALE_LLAMA_LIB like scripts/cale_dev_server.py.
Prints: seconds per probe and in total for each run; for each size in the table: the speeds of every run, their spread, and the error against the truth;
and how well the table predicts the sizes that were not measured (read off by cost, as cale/predict.py does).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--profile", default="full")
    a = ap.parse_args()
    work = Path(a.work)
    if work.exists() and any(work.iterdir()):
        sys.exit("WORK_FOLDER must be new and empty")
    work.mkdir(parents=True, exist_ok=True)
    truth = json.loads(Path(a.truth).read_text())["real"]          # id -> [prompt tokens/s, answer tokens/s]
    env = dict(os.environ, CALE_DEBUG="1", CALE_PORT="47899", PYTHONUNBUFFERED="1", CALE_NO_PATH="1")
    env.setdefault("CALE_MODEL", str(ROOT / "models" / "gemma-3n-e4b-4bit"))
    runs = []
    for i in range(a.runs):
        home = work / f"run{i + 1}"
        home.mkdir()
        t = time.time()
        r = subprocess.run([sys.executable, "-m", "cale.tune", "--profile", a.profile], cwd=ROOT, env=dict(env, CALE_HOME=str(home)), capture_output=True, text=True)
        took = time.time() - t
        try:
            rep = json.loads((home / "tune-report.json").read_text())
            pred = json.loads((home / "predict.json").read_text())
        except (OSError, ValueError):
            print(f"run {i + 1}: no result (exit {r.returncode})\n{r.stdout[-400:]}\n{r.stderr[-400:]}")
            continue
        runs.append({"wall_s": round(took, 1), "report": rep, "table": pred.get("table") or {}, "sustained": pred.get("sustained"), "noise": (pred.get("extra") or {}).get("noise"),
                     "table_prompt": pred.get("table_prompt")})
        probes = " ".join(f"{p['probe']}={p['seconds']:.0f}" for p in rep["probes"])
        print(f"run {i + 1}: whole process {took:.0f} s, tune {rep['seconds']:.0f} s   [{probes}]   noise {runs[-1]['noise']}", flush=True)
    if not runs:
        sys.exit(1)
    print(f"\ntune seconds: {[round(r['report']['seconds']) for r in runs]}   (whole process {[round(r['wall_s']) for r in runs]})")
    ids = [i for i in runs[0]["table"] if all(i in r["table"] for r in runs)]
    print(f"\n{'size':14s}{'answer tok/s per run':34s}{'spread':>8s}{'truth':>8s}{'error':>8s}   {'prompt tok/s per run':30s}")
    errs = []
    for i in ids:
        dec = [r["table"][i]["dec"] for r in runs]
        pre = [r["table"][i]["pre"] for r in runs]
        med = statistics.median(dec)
        spread = (max(dec) - min(dec)) / med if med else 0
        tr = truth.get(i)
        err = med / tr[1] - 1 if tr else None
        if err is not None:
            errs.append(abs(err))
        print(f"{i:14s}{str([round(x, 2) for x in dec]):34s}{spread * 100:>7.1f}%{(tr[1] if tr else float('nan')):>8.2f}{('' if err is None else f'{err * 100:+.1f}%'):>8s}   {[round(x, 1) for x in pre]}")
    if errs:
        print(f"\nanswer speed against the truth, at the measured sizes: mean {statistics.mean(errs) * 100:.1f}%, worst {max(errs) * 100:.1f}%")
    # the sizes that were not measured: read off by cost, as the predictor does
    from cale import gemma3n_facts as facts
    from nest.levels3n import load_levels
    levels = {l["id"]: l for l in load_levels()}
    full = max(levels.values(), key=lambda l: l["pct"])
    cost = lambda l: facts.cost(l, full)       # noqa: E731
    mid = []
    for i, tr in truth.items():
        if i in ids or i not in levels:
            continue
        pts = sorted((cost(levels[j]), statistics.median([r["table"][j]["dec"] for r in runs])) for j in ids if j in levels)
        c = cost(levels[i])
        lo = [p for p in pts if p[0] <= c]
        hi = [p for p in pts if p[0] >= c]
        if lo and hi:
            (c0, v0), (c1, v1) = lo[-1], hi[0]
            v = v0 if c1 == c0 else v0 + (v1 - v0) * (c - c0) / (c1 - c0)
            mid.append(abs(v / tr[1] - 1))
    if mid:
        print(f"the sizes in between, read off by cost: mean {statistics.mean(mid) * 100:.1f}%, worst {max(mid) * 100:.1f}%  ({len(mid)} sizes)")


if __name__ == "__main__":
    main()
