"""Step 2b: decode speed and resident RAM at each size. Run ALONE (no other GPU work).

For each size the RAM cap is set to that size, so the numbers are what a Mac at that
cap would see. Speed = median decode tok/s over short chat prompts and a math prompt, plain and with the drafter.

usage: curve_speed.py   -> results/speed/curve.json
"""
import json
import statistics
from pathlib import Path

import mlx.core as mx

from nest.io import read_jsonl
from nest.runtime import NestedGemma

SIZES = [100, 90, 80, 70, 60, 50]
PROMPTS = {"chat": [r for r in read_jsonl("data/bench/ifeval.jsonl")[46:50]],
           "math": [r for r in read_jsonl("data/bench/gsm8k.jsonl")[53:55]]}
TOKENS = {"chat": 120, "math": 160}

mx.set_cache_limit(1024 ** 3)
M = NestedGemma.load("models/g12-nest/model.nest")
out = {}
M.generate(PROMPTS["chat"][0], max_tokens=16, width=1.0, draft=False)          # warm-up
for pct in SIZES:
    M.set_cap(pct / 100)
    mx.clear_cache()
    row = {"resident_gb": round(mx.get_active_memory() / 1e9, 2), "gb_per_pass": round(M.width_gb(pct / 100), 2)}
    for draft in (False, True):
        for kind, recs in PROMPTS.items():
            tps = []
            for r in recs:
                st = M.generate(r, max_tokens=TOKENS[kind], width=pct / 100, draft=draft)["stats"]
                if st.get("tok_per_s"):
                    tps.append(st["tok_per_s"])
            row[f"{kind}{'_draft' if draft else ''}_tps"] = round(statistics.median(tps), 2) if tps else None
    out[str(pct)] = row
    print(pct, row, flush=True)
Path("results/speed").mkdir(parents=True, exist_ok=True)
Path("results/speed/curve.json").write_text(json.dumps(out, indent=1))
