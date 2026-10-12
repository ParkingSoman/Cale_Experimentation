"""Gate checks for the llama.cpp backend. One run = one library + one model + one size.

usage: llama_gates.py --model M.gguf --out out.json [--lib LIB] [--size e4b|e2b|<level id>] [--n 12] [--gen 6]
Writes: greedy token ids for a few prompts, and the mean loss (nats) on the answer part of held-out chat samples.
Compare runs with: llama_gates.py --compare a.json b.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROMPTS = ["Write a short, friendly email to a neighbour asking them to keep the noise down after 10pm.",
           "What is 17 times 23? Explain the steps.",
           "List three facts about the ocean.",
           "Explain what a hash table is to a ten year old.",
           "Translate to French: The library opens at nine in the morning.",
           "Give me a haiku about autumn rain."]


def widths_for(level, n_layers=35):
    return [0 if i in level["skip"] else level["blocks"][i] * 8192 for i in range(n_layers)]


def run(a):
    from cale.backends.llamalib import Lib
    from nest.levels3n import load_levels
    t0 = time.perf_counter()
    lib = Lib(a.model, n_threads=a.threads, n_ctx=4096, lib_path=a.lib)
    load_s = time.perf_counter() - t0
    if a.size not in ("none", "file"):
        lv = next(l for l in load_levels(lib.n_layers) if l["id"] == a.size)
        lib.set_widths(widths_for(lv, lib.n_layers))
    out = {"model": Path(a.model).name, "size": a.size, "lib": a.lib, "load_s": round(load_s, 1), "gen": [], "nll": []}
    for p in PROMPTS[:a.gen]:
        text = lib.chat_text([{"role": "user", "content": p}])
        ids = lib.tokenize(text)
        t = time.perf_counter()
        got = lib.generate_ids(ids, a.gen_tokens)
        out["gen"].append({"prompt_tokens": len(ids), "ids": got, "seconds": round(time.perf_counter() - t, 2)})
        print("gen", len(got), "tokens", round(time.perf_counter() - t, 1), "s", flush=True)
    rows = [json.loads(l) for l in (ROOT / "data" / "heldout.jsonl").read_text().splitlines()]
    rows = [r for r in rows if not r.get("image") and r.get("assistant")][:a.n]
    for r in rows:
        text = lib.chat_text([{"role": "user", "content": r["user"]}])
        p = lib.tokenize(text)[-a.ptrunc:]
        ans = lib.tokenize(r["assistant"], add_special=False)[:a.atrunc]
        out["nll"].append(lib.nll(p + ans, first=len(p)))
        print("nll", len(out["nll"]), round(out["nll"][-1], 4), flush=True)
    out["mean_nll"] = sum(out["nll"]) / max(len(out["nll"]), 1)
    Path(a.out).write_text(json.dumps(out))
    print("GATES_DONE mean_nll", round(out["mean_nll"], 4), flush=True)


def compare(x, y):
    A, B = (json.loads(Path(p).read_text()) for p in (x, y))
    same = sum(1 for g, h in zip(A["gen"], B["gen"]) if g["ids"] == h["ids"])
    print(f"greedy answers identical: {same}/{len(A['gen'])}")
    d = [abs(p - q) for p, q in zip(A["nll"], B["nll"])]
    print(f"mean loss {A['mean_nll']:.4f} vs {B['mean_nll']:.4f}; largest per-sample difference {max(d) if d else 0:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--lib")
    ap.add_argument("--size", default="none")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--gen", type=int, default=6)
    ap.add_argument("--gen-tokens", dest="gen_tokens", type=int, default=24)
    ap.add_argument("--ptrunc", type=int, default=300)
    ap.add_argument("--atrunc", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    compare(*a.compare) if a.compare else run(a)
