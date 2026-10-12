"""Decode speed per tier AND per content type, round-robin, with E4B in the same rounds.

The drafter's speed depends on the text: on instruction-following prompts at 85% width it
reached 10-18 tok/s, vs 22 on GSM8K/chat prompts. So the mixed-workload simulator needs
speeds measured on matching content.

Prompts (not used anywhere else): IFEval 46-57 ("easy"), GSM8K 53-58 ("hard_long").
usage: content_speed.py [--reps 2]   -> results/speed/content.json
"""
import argparse
import json
import statistics
import time

import mlx.core as mx

from nest.gen import EOS, _prefill
from nest.io import encode, load, read_jsonl
from nest.runtime import NestedGemma

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=2)
args = ap.parse_args()

PROMPTS = {"easy": (read_jsonl("data/bench/ifeval.jsonl")[46:58], 200),
           "hard_long": (read_jsonl("data/bench/gsm8k.jsonl")[53:59], 250)}
M = NestedGemma.load("models/g12-nest/model.nest")
E4B, E4B_PROC = load("models/gemma-4-e4b-it-4bit")


def e4b_generate(rec, max_tokens):
    ids, kw = encode(E4B_PROC, rec, with_answer=False, max_tokens=4096)
    lm = E4B.language_model
    cache = lm.make_cache()
    t0 = time.perf_counter()
    h = _prefill(E4B, ids, kw, cache)
    tok = mx.argmax(lm.logits_from_hidden(h[:, -1:]), axis=-1)
    n = 0
    for _ in range(max_tokens):
        if int(tok.item()) in EOS:
            break
        n += 1
        h = lm.model(tok.reshape(1, 1), cache=cache)
        tok = mx.argmax(lm.logits_from_hidden(h[:, -1:]), axis=-1)
        mx.async_eval(tok)
    return {"prompt_tokens": int(ids.shape[1]), "tokens": n, "seconds": time.perf_counter() - t0}


TIERS = {"w100": (1.0, False), "w85": (0.85, False), "w100+draft": (1.0, True), "w85+draft": (0.85, True), "E4B": None}
rows = {(t, c): [] for t in TIERS for c in PROMPTS}
for rep in range(args.reps + 1):                   # round 0 is warm-up
    for tier, cfg in TIERS.items():
        for kind, (recs, mt) in PROMPTS.items():
            for r in (recs if rep else recs[:1]):
                if cfg is None:
                    st = e4b_generate(r, mt)
                else:
                    st = M.generate(r, max_tokens=mt, width=cfg[0], draft=cfg[1])["stats"]
                if rep:
                    rows[(tier, kind)].append((st["tokens"], st["seconds"], st["prompt_tokens"]))
        if rep:
            print(rep, tier, {k: round(sum(x[0] for x in rows[(tier, k)]) / sum(x[1] for x in rows[(tier, k)]), 2)
                              for k in PROMPTS}, flush=True)

# Effective speed = generated tokens / wall seconds of whole calls (prefill included), per call type.
out = {}
for (tier, kind), xs in rows.items():
    out.setdefault(tier, {})[kind] = {"effective_tps": sum(x[0] for x in xs) / sum(x[1] for x in xs),
                                      "median_call_s": statistics.median(x[1] for x in xs),
                                      "mean_tokens": statistics.mean(x[0] for x in xs)}
json.dump(out, open("results/speed/content.json", "w"), indent=1)
print(json.dumps(out, indent=1))
