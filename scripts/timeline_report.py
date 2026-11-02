"""Turn timeline.jsonl (cale/timeline.py) into a table: seconds since the start, and seconds since the step before.

usage: python scripts/timeline_report.py [FILE ...] [--anchor chat_start] [--last N]
  FILE      timeline.jsonl files. Default: the one in the Cale folder (CALE_HOME). Give one per computer to put them side by side.
  --anchor  the step that starts a run. Default chat_start. Parts of one run are separate processes, so runs are cut by time, not by run id.
  --last N  show the newest N runs (default 1)
With two or more files the table shows each computer in a column and the gap between the first and the last column.
Then it judges the moments a person feels against the sameness gate (see GATE below) and says PASS or FAIL for each.
"""
import argparse
import json
import os
import sys
from pathlib import Path


# The sameness gate (plan of 2026-10-18; a first proposal, to be set from real tables): a moment passes when the slower computer is within
# 1 second of the faster one, or within 1.5 times of it. Feeling the same matters more than a number of seconds.
GATE_SECONDS, GATE_RATIO = 1.0, 1.5
MOMENTS = (("Cale reachable (server answers)", "cale_reachable"), ("Chat ready (a person can type)", "ready_said"),
           ("AI ready (first word will not wait)", "model_object_ready"))


def judge(a, b):
    """(gap seconds, ratio, passed) for one moment on two computers."""
    lo, hi = min(a, b), max(a, b)
    gap = hi - lo
    ratio = hi / lo if lo > 0 else float("inf")
    return gap, ratio, (gap <= GATE_SECONDS or ratio <= GATE_RATIO)


def default_file():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cale import paths
    return paths.HOME / "timeline.jsonl"


def load(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("kind") == "mark" and "step" in d:
            rows.append(d)
    return sorted(rows, key=lambda r: r["t"])


def runs(rows, anchor):
    """Cut the marks into runs. A run starts at each `anchor` mark. Marks from before the first anchor belong to the first run (the server may start earlier)."""
    cuts = [i for i, r in enumerate(rows) if r["step"] == anchor]
    if not cuts:
        return []
    out = []
    for n, i in enumerate(cuts):
        end = cuts[n + 1] if n + 1 < len(cuts) else len(rows)
        out.append(rows[i:end])
    return out


def table(run, anchor):
    t0 = next(r["t"] for r in run if r["step"] == anchor)
    prev, out = t0, []
    for r in run:
        out.append((r["step"], round(r["t"] - t0, 1), round(r["t"] - prev, 1), r.get("seconds")))
        prev = r["t"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--anchor", default="chat_start")
    ap.add_argument("--last", type=int, default=1)
    a = ap.parse_args()
    files = a.files or [str(default_file())]
    per = []
    for f in files:
        rs = runs(load(f), a.anchor)
        if not rs:
            print(f"{f}: no '{a.anchor}' mark")
            return 1
        per.append(rs[-a.last:] if a.last > 1 else [rs[-1]])
    if len(files) == 1:
        for run in per[0]:
            print(f"{'step':24s} {'since start':>12s} {'since previous':>15s}")
            for step, since, delta, secs in table(run, a.anchor):
                print(f"{step:24s} {since:>11.1f}s {delta:>14.1f}s" + (f"   ({secs} s)" if secs is not None else ""))
            print()
        return 0
    cols = [{s: since for s, since, _, _ in table(rs[-1], a.anchor)} for rs in per]
    steps = []
    for c in cols:
        for s in c:
            if s not in steps:
                steps.append(s)
    print(f"{'step':24s}" + "".join(f"{Path(f).parent.name[:12]:>14s}" for f in files) + f"{'gap':>8s}")
    for s in steps:
        vals = [c.get(s) for c in cols]
        gap = round(vals[-1] - vals[0], 1) if vals[0] is not None and vals[-1] is not None else None
        print(f"{s:24s}" + "".join(f"{'-' if v is None else f'{v:.1f}s':>14s}" for v in vals) + f"{'' if gap is None else f'{gap:+.1f}s':>8s}")
    print(f"\nSameness gate: within {GATE_SECONDS:g} s or {GATE_RATIO:g}x of the other computer ({Path(files[0]).parent.name} against {Path(files[-1]).parent.name})")
    for label, step in MOMENTS:
        a, b = cols[0].get(step), cols[-1].get(step)
        if a is None or b is None:
            print(f"  {label:38s} not in both timelines")
            continue
        gap, ratio, ok = judge(a, b)
        print(f"  {label:38s} {a:5.1f}s {b:5.1f}s  gap {gap:4.1f}s  {ratio:4.1f}x  {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
