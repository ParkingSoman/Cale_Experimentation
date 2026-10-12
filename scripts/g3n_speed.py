"""Speed of every Gemma 3n size on one loaded model. Run it when nothing else uses the GPU.

usage: g3n_speed.py MODEL_DIR OUT.json [--runs 3]
Per size: median generation tok/s over `runs` runs of 128 tokens on a fixed prompt, median switch time (ms),
and the memory the model holds. Sizes are visited in a round-robin so a busy GPU hits all sizes alike.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PROMPT = "Write a short, friendly email to a neighbour asking them to keep the noise down after 10pm."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    from nest.gemma3n import NestedGemma3n
    m = NestedGemma3n.load(a.model)
    m.generate(PROMPT, 16)                                    # warm up
    tps, sw = {l["id"]: [] for l in m.levels}, {l["id"]: [] for l in m.levels}
    for _ in range(a.runs):
        for lv in m.levels:
            sw[lv["id"]].append(m.set(lv["id"]))
            r = m.generate(PROMPT, 128)["stats"]
            tps[lv["id"]].append(r["tokens"] / r["seconds"])
    out = {"runs": a.runs, "active_gb": round(mx.get_active_memory() / 1e9, 2),
           "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
           "sizes": {k: {"tok_per_s": round(statistics.median(v), 1), "switch_ms": round(statistics.median(sw[k]), 3)}
                     for k, v in tps.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in out["sizes"].items():
        print(f'{k:14} {v["tok_per_s"]:6.1f} tok/s   switch {v["switch_ms"]:.3f} ms')
    print("active", out["active_gb"], "GB  peak", out["peak_gb"], "GB")


if __name__ == "__main__":
    main()
