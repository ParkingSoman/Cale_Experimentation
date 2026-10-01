"""Measure prefill and decode speed per level, plus the bytes each level uses.

usage: speed.py MODEL_DIR NAME [--nest] [--level LEVELS_JSON] [--reps 5]
Writes results/speed/NAME.json
"""
import argparse
import json
import os
import statistics
import time

import mlx.core as mx
from mlx.utils import tree_flatten

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("name")
ap.add_argument("--nest", action="store_true")
ap.add_argument("--level")
ap.add_argument("--reps", type=int, default=9)
ap.add_argument("--prompt-tokens", type=int, default=512)
ap.add_argument("--gen-tokens", type=int, default=128)
args = ap.parse_args()

if args.nest:
    from nest import elastic
    elastic.patch()
    elastic.ElasticMLP.LEAN = False
from nest.io import load

model, proc = load(args.model)
lm = model.language_model


def level_bytes(widths=None, skip=()):
    """Bytes of the weights a level touches (language model + embedders)."""
    skip = set(skip)
    total = 0
    for k, v in tree_flatten(model.parameters()):
        if ".layers." in k:
            i = int(k.split(".layers.")[1].split(".")[0])
            if i in skip:
                continue
            if ".mlp." in k and widths is not None:
                total += v.nbytes * widths[i] / 15360          # share of neurons this level uses
                continue
        total += v.nbytes
    return total


def one_run(prompt_ids):
    cache = lm.make_cache()
    t0 = time.perf_counter()
    h = lm.model(prompt_ids, cache=cache)
    tok = mx.argmax(lm.logits_from_hidden(h[:, -1:]), axis=-1)
    mx.eval(tok)
    t1 = time.perf_counter()
    for _ in range(args.gen_tokens):
        h = lm.model(tok.reshape(1, 1), cache=cache)
        tok = mx.argmax(lm.logits_from_hidden(h[:, -1:]), axis=-1)
        mx.async_eval(tok)
    mx.eval(tok)
    t2 = time.perf_counter()
    return prompt_ids.shape[1] / (t1 - t0), args.gen_tokens / (t2 - t1)


levels = json.load(open(args.level)) if args.level else {"full": None}
prompt = mx.random.randint(1000, 200000, (1, args.prompt_tokens), key=mx.random.key(0))


def set_lv(lv):
    if args.nest:
        widths, skip = (None, ()) if lv is None else (lv.get("widths"), lv.get("skip", ()))
        elastic.set_level(model, widths, skip)


# Round-robin: every round runs every level once, so a slowdown from other GPU users
# hits all levels of that round. Report medians and per-round ratios to the first level.
names = list(levels)
for n in names:            # warm-up
    set_lv(levels[n]); one_run(prompt)
runs = {n: [] for n in names}
for r in range(args.reps):
    for n in names:
        set_lv(levels[n])
        runs[n].append(one_run(prompt))
base = names[0]
out = {}
for n in names:
    lv = levels[n]
    widths, skip = (None, ()) if lv is None else (lv.get("widths"), lv.get("skip", ()))
    pre = statistics.median(x[0] for x in runs[n])
    dec = statistics.median(x[1] for x in runs[n])
    ratio = statistics.median(runs[n][i][1] / runs[base][i][1] for i in range(args.reps))
    pratio = statistics.median(runs[n][i][0] / runs[base][i][0] for i in range(args.reps))
    gb = level_bytes(widths, skip) / 1e9
    out[n] = {"prefill_tps": pre, "decode_tps": dec, "decode_vs_first": ratio, "prefill_vs_first": pratio,
              "level_gb": gb, "decode_runs": [x[1] for x in runs[n]]}
    print(f"[{args.name}/{n}] prefill {pre:7.1f} tok/s (x{pratio:.2f}) | decode {dec:6.1f} tok/s (x{ratio:.2f}) | level weights {gb:.2f} GB", flush=True)
out["_peak_gb"] = mx.get_peak_memory() / 1e9
os.makedirs("results/speed", exist_ok=True)
json.dump(out, open(f"results/speed/{args.name}.json", "w"), indent=1)
print("peak GB", out["_peak_gb"])
