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
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--prompt-tokens", type=int, default=512)
ap.add_argument("--gen-tokens", type=int, default=128)
args = ap.parse_args()

if args.nest:
    from nest import elastic
    elastic.patch()
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
                total += v.nbytes * widths[i] / v.shape[0]   # nest layout: rows are neurons
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
out = {}
for lname, lv in levels.items():
    widths, skip = (None, ()) if lv is None else (lv.get("widths"), lv.get("skip", ()))
    if args.nest:
        elastic.set_level(model, widths, skip)
    one_run(prompt)  # warm-up
    runs = [one_run(prompt) for _ in range(args.reps)]
    pre = statistics.median(r[0] for r in runs)
    dec = statistics.median(r[1] for r in runs)
    gb = level_bytes(widths, skip) / 1e9
    out[lname] = {"prefill_tps": pre, "decode_tps": dec, "level_gb": gb,
                  "decode_runs": [r[1] for r in runs]}
    print(f"[{args.name}/{lname}] prefill {pre:7.1f} tok/s | decode {dec:6.1f} tok/s | level weights {gb:.2f} GB", flush=True)
out["_peak_gb"] = mx.get_peak_memory() / 1e9
os.makedirs("results/speed", exist_ok=True)
json.dump(out, open(f"results/speed/{args.name}.json", "w"), indent=1)
print("peak GB", out["_peak_gb"])
