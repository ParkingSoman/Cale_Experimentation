"""Speed of Google's QAT MTP drafter on top of nest levels, vs. the same level without it.

Round-robin over (level, with/without drafter) so GPU contention hits all cells alike.
Greedy decoding, text prompts. Reports median generation tok/s, speedup, and the
drafter's acceptance stats, plus whether drafted output matches plain greedy output.
"""
import json
import statistics
import sys

import mlx.core as mx

from nest import elastic
elastic.patch()
elastic.ElasticMLP.LEAN = False
from mlx_vlm import generate
from mlx_vlm.speculative.drafters import load_drafter
from mlx_vlm.speculative.utils import format_speculative_stats

from nest.io import load, read_jsonl

MODEL, DRAFTER = "models/qg4", "models/qat-assistant-4bit"
LEVELS = json.load(open("results/levels/speed1.json"))
LEVELS = {k: LEVELS[k] for k in ("full", "u90", "u75")}
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

model, proc = load(MODEL)
drafter, kind = load_drafter(DRAFTER)
print("drafter kind:", kind, flush=True)
prompts = [r["user"] for r in read_jsonl("data/bench/gsm8k.jsonl")[:3]]
prompts += [r["user"] for r in read_jsonl("data/heldout.jsonl") if r["source"] in ("chat", "code")][:3]
texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": p}]}],
                                  tokenize=False, add_generation_prompt=True) for p in prompts]

tps = {}
same = {}
for rep in range(REPS):
    for lname, lv in LEVELS.items():
        elastic.set_level(model, None if lv is None else lv["widths"])
        for use in (False, True):
            vals, outs = [], []
            for t in texts:
                kw = dict(draft_model=drafter, draft_kind=kind) if use else {}
                r = generate(model, proc, t, max_tokens=160, temperature=0.0, verbose=False, **kw)
                vals.append(r.generation_tps)
                outs.append(r.text)
            tps.setdefault((lname, use), []).append(statistics.median(vals))
            if rep == 0:
                same.setdefault(lname, {})[use] = outs
            if use and rep == REPS - 1:
                print(f"[{lname}] drafter stats: {format_speculative_stats(drafter)}", flush=True)
        print(f"round {rep} {lname}: plain {tps[(lname, False)][-1]:.1f} tok/s, drafter {tps[(lname, True)][-1]:.1f} tok/s", flush=True)

print("\nlevel | plain tok/s | with drafter tok/s | speedup (median of per-round ratios) | outputs identical")
for lname in LEVELS:
    p, d = tps[(lname, False)], tps[(lname, True)]
    ratio = statistics.median(d[i] / p[i] for i in range(REPS))
    ident = sum(a == b for a, b in zip(same[lname][False], same[lname][True]))
    print(f"{lname:5s} | {statistics.median(p):6.1f} | {statistics.median(d):6.1f} | x{ratio:.2f} | {ident}/{len(texts)}")
