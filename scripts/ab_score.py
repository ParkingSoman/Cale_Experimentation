#!/usr/bin/env python3
"""Combine the A/B results (feel) with the model's measured quality into one score per way of asking.

For each call: score = quality of the size that ran, if the call met the feel target, else 0. A way's score is the mean over its calls.
  quality = share of the full size's next-word choices this size repeats (cale/gemma3n_quality.json, measured on held-out text).
  1.00 means: every call on target AND the full model's quality. The ceiling is the best quality any size could have while on target.

usage: python3 scripts/ab_score.py [--out results/demo/ab-combined.json]      reads results/demo/ab-{mac,hp-windows,hp-linux}.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cale import gemma3n_facts as facts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "results/demo/ab-combined.json"))
a = ap.parse_args()
LEVELS = facts.levels()


def quality_of(size_pct):
    lvl = min(LEVELS, key=lambda l: abs(l["pct"] - size_pct))
    return facts.quality(lvl)


out = {"note": "score = mean over calls of (quality of the size that ran, if the call met first word <= 1 s and >= 8 tok/s, else 0)", "computers": {}}
for name in ("mac", "hp-windows", "hp-linux"):
    d = json.load(open(ROOT / f"results/demo/ab-{name}.json"))
    t = d["target"]
    res = {}
    for arm, v in d["arms"].items():
        scores, on = [], 0
        for r in v["rows"]:
            hit = r["first_s"] is not None and r["first_s"] <= t["first_s"] and r["tps"] is not None and r["tps"] >= t["tps"]
            on += hit
            scores.append(quality_of(r["size"]) if hit and r["size"] else 0.0)
        res[arm] = {"score": round(sum(scores) / len(scores), 3), "on_target": on, "calls": len(scores), "size": v["size_median"],
                    "quality": round(quality_of(v["size_median"]), 3)}
    out["computers"][name] = res
    print(f"{name:11}" + "".join(f"  {arm}: {r['score']:.2f} ({r['on_target']}/{r['calls']} on target, quality {r['quality']:.2f})" for arm, r in res.items()))
Path(a.out).write_text(json.dumps(out, indent=1))
print("saved", a.out)
