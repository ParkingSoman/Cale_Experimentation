"""Step 6c check: run a real mixed workload (16 easy + 4 hard calls, developer-routed) through
the runtime and compare its wall time with the simulator's formula on the same calls.

Easy = IFEval prompts 31-46 (not in the scored 30). Hard = 2 GSM8K (51-52) + 2 MMLU-Pro one-pass.
Pass: predicted total within 10% of measured.

usage: mix_direct.py   -> results/mix-direct.json
"""
import json
import random
import time

from nest.io import read_jsonl
from nest.runtime import NestedGemma

SP = json.load(open("results/speed/nest.json"))
calls = [("easy", r) for r in read_jsonl("data/bench/ifeval.jsonl")[30:46]]
calls += [("hard_long", r) for r in read_jsonl("data/bench/gsm8k.jsonl")[50:52]]
calls += [("hard_one", r) for r in read_jsonl("data/bench/mmlupro1k.jsonl")[:2]]
random.Random(0).shuffle(calls)
ROUTE = {"easy": ("w85+draft", 0.85, True, 200), "hard_long": ("w100+draft", 1.0, True, 250),
         "hard_one": ("w100", 1.0, False, 0)}

M = NestedGemma.load("models/g12-nest/model.nest")
for w, d in ((1.0, True), (0.85, True), (1.0, False)):      # warm every tier the run uses
    M.generate("warm up", max_tokens=8, width=w, draft=d)
rows, t0 = [], time.perf_counter()
for kind, r in calls:
    tier, w, d, max_out = ROUTE[kind]
    out = M.choose(r, width=w) if kind == "hard_one" else M.generate(r, max_tokens=max_out, width=w, draft=d)
    st = out["stats"]
    pre, dec = SP[tier]["prefill_tps"], SP[tier]["decode_tps"]
    pred = st["prompt_tokens"] / pre + st.get("tokens", 0) / dec
    rows.append({"kind": kind, "tier": st["tier"], "seconds": st["seconds"], "predicted_s": pred,
                 "prompt_tokens": st["prompt_tokens"], "tokens": st.get("tokens", 0), "switch_ms": st["switch_ms"]})
total = time.perf_counter() - t0
pred = sum(r["predicted_s"] for r in rows)
res = {"measured_total_s": total, "predicted_total_s": pred, "error": (pred - total) / total, "calls": rows}
json.dump(res, open("results/mix-direct.json", "w"), indent=1)
print(f"measured {total:.1f}s, predicted {pred:.1f}s, error {100 * res['error']:+.1f}%")
