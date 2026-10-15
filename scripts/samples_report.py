#!/usr/bin/env python3
"""What do the passive samples say? Speed by how crowded and how busy the computer was. Numbers only.

usage: python3 scripts/samples_report.py [SAMPLES_FILE ...]      (default: Cale's own samples.jsonl)
It prints, for each (where, size), the median answer speed alone and under load, and the slowdown. That table is the data a load model needs:
the predictor today assumes other programs cost at most 0.5x (cale/predict.py live_factor).
"""
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load(paths):
    rows = []
    for p in paths:
        for l in Path(p).read_text().splitlines():
            try:
                rows.append(json.loads(l))
            except ValueError:
                pass
    return rows


def bucket_crowd(c):
    return "alone" if c is None or c < 1.05 else ("2" if c < 2.5 else "3+")


def bucket_busy(b):
    return "quiet" if b is None or b < 0.15 else ("busy" if b < 0.5 else "very busy")


def main():
    default = Path(os.environ.get("CALE_HOME", Path.home() / ".local/share/cale")) / "samples.jsonl"
    rows = load([a for a in sys.argv[1:]] or [default])
    calls = [r for r in rows if r.get("kind") == "call" and r.get("tps") and r.get("tokens", 0) >= 9]
    print(f"{len(rows)} samples, {len(calls)} calls with a speed (9 or more tokens)")
    groups = defaultdict(list)
    for r in calls:
        groups[(r.get("where", "?"), r.get("size"), bucket_crowd(r.get("crowd")), bucket_busy(r.get("busy")))].append(r["tps"])
    base = {}
    for (where, size, crowd, busy), v in groups.items():
        if crowd == "alone" and busy == "quiet":
            base[(where, size)] = statistics.median(v)
    print(f"{'where':7s}{'size%':>6s}{'crowd':>8s}{'busy':>11s}{'calls':>7s}{'median tok/s':>14s}{'vs quiet':>10s}")
    for key in sorted(groups, key=lambda k: (k[0], k[1] or 0, k[2], k[3])):
        v = groups[key]
        med = statistics.median(v)
        b = base.get(key[:2])
        print(f"{key[0]:7s}{str(key[1]):>6s}{key[2]:>8s}{key[3]:>11s}{len(v):>7d}{med:>14.2f}{(med / b if b else float('nan')):>10.2f}")
    cal = [r for r in rows if r.get("kind") == "call" and r.get("tps") and r.get("predicted_tps")]
    if cal:
        ratios = sorted(r["tps"] / r["predicted_tps"] for r in cal)
        pick = lambda q: ratios[min(len(ratios) - 1, int(q * len(ratios)))]          # noqa: E731
        print(f"\ncalibration (actual / predicted speed, 1.0 is exact): median {statistics.median(ratios):.2f}, "
              f"10th {pick(0.1):.2f}, 90th {pick(0.9):.2f} over {len(cal)} answers  (above 1: too cautious, below 1: too hopeful)")
    heads = [r for r in rows if r.get("kind") == "call" and r.get("headroom")]
    if heads:
        wasted = sum(1 for r in heads if r.get("wasted"))
        print(f"headroom over the goals: median {statistics.median(r['headroom'] for r in heads):.2f}; "
              f"{wasted} of {len(heads)} answers beat every goal by over 30% while a larger size was allowed (wasted slice)")
    stalls = [r for r in rows if r.get("kind") == "call" and r.get("step_max") and r.get("step_p50")]
    if stalls:
        ratio = [r["step_max"] / r["step_p50"] for r in stalls]
        print(f"\nstalls: worst step is {statistics.median(ratio):.1f}x the median step (median over {len(stalls)} answers); "
              f"{sum(1 for x in ratio if x > 5)} answers had a step over 5x the median")
    paused = [r for r in rows if r.get("kind") == "call" and (r.get("paused_dec_s") or 0) > 0]
    print(f"answers that were paused for another app: {len(paused)}")
    host = [r for r in rows if r.get("kind") == "host"]
    if host:
        print(f"host looks: {len(host)}, pressure levels seen: {sorted({r.get('pressure') for r in host if r.get('pressure') is not None})}")


if __name__ == "__main__":
    main()
