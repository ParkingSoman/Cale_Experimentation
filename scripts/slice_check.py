"""Step 1 checks for any-size slicing. Run twice, then compare:

  PYTHONPATH=. .venv/bin/python scripts/slice_check.py blocks   # old path (RAM budget set)
  PYTHONPATH=. .venv/bin/python scripts/slice_check.py any      # joined tables, any width + cap

`any` also tests odd widths, per-layer widths and set_cap shrink/grow.
Results go to results/slice-check-<mode>.json.
"""
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from nest.runtime import NestedGemma

MODEL = "models/g12-nest/model.nest"
PROMPTS = ["Explain in two sentences why the sky is blue.",
           "List three differences between a list and a tuple in Python.",
           "What is 17 * 24? Show the steps briefly."]
N = 40


def gb():
    return round(mx.get_active_memory() / 1e9, 2)


def run(m, width):
    outs = [m.generate(p, max_tokens=N, width=width, draft=False) for p in PROMPTS]
    return {"text": [o["text"] for o in outs], "tok_per_s": [o["stats"]["tok_per_s"] for o in outs],
            "tier": outs[0]["stats"]["tier"]}


def main(mode):
    res = {}
    if mode == "blocks":
        m = NestedGemma.load(MODEL, ram_budget_gb=64, drafter=False)   # budget big enough to hold all
        for w in (1.0, 0.9, 0.85):
            res[f"w{w}"] = run(m, w)
    else:
        m = NestedGemma.load(MODEL, drafter=False)
        res["mem_loaded_gb"] = gb()
        for w in (1.0, 0.9, 0.85):
            res[f"w{w}"] = run(m, w)
        for w in (0.5, 0.63, 0.77):                                    # widths the old file could not run
            t = time.perf_counter()
            sw = m.set(width=w)
            res[f"w{w}"] = run(m, w)
            res[f"w{w}"]["set_ms"] = round(sw, 1)
        layers = len(m.layers)
        res["perlayer"] = run(m, [1.0 if i % 2 == 0 else 0.6 for i in range(layers)])
        m.set(width=1.0)
        # cap: shrink to 0.6 -> memory drops, text at 0.6 unchanged; grow back -> full text unchanged
        before = res["w0.63"] if False else run(m, 0.6)
        mem0 = gb()
        t = time.perf_counter()
        m.set_cap(0.6)
        res["cap_shrink_s"] = round(time.perf_counter() - t, 2)
        res["cap_mem_gb"] = [mem0, gb()]
        res["cap_req_1.0"] = run(m, 1.0)                                # clamped to the cap
        res["cap_same_text_at_0.6"] = run(m, 0.6)["text"] == before["text"]
        t = time.perf_counter()
        m.set_cap(1.0)
        res["cap_grow_s"] = round(time.perf_counter() - t, 2)
        res["cap_mem_after_grow_gb"] = gb()
        res["cap_full_same_text"] = run(m, 1.0)["text"] == res["w1.0"]["text"]
    Path("results").mkdir(exist_ok=True)
    Path(f"results/slice-check-{mode}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else v.get("tier")) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
