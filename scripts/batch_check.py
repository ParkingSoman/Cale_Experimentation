"""Do answers that run together equal the same answers run alone, and how much faster is the whole? Needs a llama.cpp engine library with slots.

usage: python scripts/batch_check.py MODEL.cale.gguf [--lib LIB] [--tokens 32] [--sizes 1,2,4]
For each number of answers n: n different prompts run together. Prints each answer's speed, the total speed, and whether each answer is
the same text as when it ran alone (greedy). Small differences can appear: a different batch changes the order of additions.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
PROMPTS = ["Name three planets and one fact about each.", "Write two sentences about the sea.", "List four fruits with their colors.",
           "Explain in two sentences why the sky is blue."]


def run(m, prompts, tokens):
    streams = [m.open_stream([{"role": "user", "content": p}], tokens, 0.0, 1.0, 1, None, None, f"app{i}") for i, p in enumerate(prompts)]
    t0, text, live = time.perf_counter(), {s.slot: "" for s in streams}, list(streams)
    firsts = {}
    while live:
        done = m.step(live, 128)
        for s in live:
            text[s.slot] += "".join(s.take())
            if s.t_first is not None and s.slot not in firsts:
                firsts[s.slot] = time.perf_counter()
        for s in done:
            live.remove(s)
    total = time.perf_counter() - t0
    out = [(text[s.slot], s.out, s) for s in streams]
    return out, total, firsts, t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lib")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--sizes", default="1,2,4")
    a = ap.parse_args()
    os.environ.setdefault("CALE_SLOTS", "4")
    os.environ.setdefault("CALE_GPU", "off")
    from cale.backends.llama import LlamaNested
    m = LlamaNested.load(a.model, lib_path=a.lib)
    print(f"slots: {m.n_slots}, placement: {m.placement}", flush=True)
    alone = {}
    for p in PROMPTS:
        out, total, firsts, t0 = run(m, [p], a.tokens)
        alone[p] = out[0][0]
    solo_tps = None
    for n in [int(x) for x in a.sizes.split(",")]:
        prompts = PROMPTS[:n]
        out, total, firsts, t0 = run(m, prompts, a.tokens)
        gen = sum(o[1] for o in out)
        decode_s = total - max(firsts.values()) + t0 if firsts else total
        same = sum(1 for p, o in zip(prompts, out) if o[0] == alone[p])
        per = [(o[1] - 1) / max(o[2].t_first and (time.perf_counter() - o[2].t_first) or 1, 1e-6) for o in out]
        print(f"{n} together: {gen} tokens in {total:.1f} s = {gen / total:.1f} tokens/s in all; same text as alone: {same}/{n}", flush=True)
    m.close()


if __name__ == "__main__":
    main()
