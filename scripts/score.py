"""Score one model (and optionally one nest level) against the bf16 reference.

usage: score.py MODEL_DIR NAME [--nest] [--level LEVEL_JSON] [--limit N]
Writes results/scores/NAME.json
"""
import argparse
import json
import os
import time

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("name")
ap.add_argument("--nest", action="store_true", help="model uses the nest layout (patch before load)")
ap.add_argument("--level", help="JSON file with {widths, skip} or a list of such levels")
ap.add_argument("--per-source", type=int, default=25, help="samples per source")
ap.add_argument("--ref", default="results/ref-bf16")
args = ap.parse_args()

if args.nest:
    from nest import elastic
    elastic.patch()
from nest.drift import fmt, score
from nest.io import load, read_jsonl

recs = read_jsonl("data/heldout.jsonl")
count = {}
sel = []
for r in recs:
    count[r["source"]] = count.get(r["source"], 0) + 1
    if count[r["source"]] <= args.per_source:
        sel.append(r)

mx.set_cache_limit(2 * 1024**3)
model, proc = load(args.model)
levels = {"full": None}
if args.level:
    levels = json.load(open(args.level))
os.makedirs("results/scores", exist_ok=True)
out = {}
for lname, lv in levels.items():
    if args.nest:
        if lv is None:
            elastic.set_level(model)
        else:
            elastic.set_level(model, lv.get("widths"), lv.get("skip", ()))
    t = time.time()
    res = score(model, proc, sel, args.ref)
    print(f"[{args.name}/{lname}] {time.time() - t:.0f}s  {fmt(res)}", flush=True)
    res["_seconds"] = time.time() - t
    out[lname] = res
    json.dump(out, open(f"results/scores/{args.name}.json", "w"), indent=1)
json.dump(out, open(f"results/scores/{args.name}.json", "w"), indent=1)
print("peak GB", mx.get_peak_memory() / 1e9)
