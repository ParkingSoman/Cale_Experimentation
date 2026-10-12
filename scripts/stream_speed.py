"""Step 6b: speed per tier of the one-file 12B (width x drafter), round-robin, medians.

Per tier:
  prefill tok/s and time-to-answer for a one-pass call (512-token prompt)
  decode tok/s on real prompts (3 GSM8K + 3 chat/code, greedy, --gen tokens)
  SSD GB read per call (non-zero only when --ram keeps part of the model on disk)
For a tier that streams, two controls on the prefill pass:
  compute-only  same math, streamed blocks read into RAM beforehand (no reads in the pass)
  reads-only    read the streamed bytes of one pass, no compute

usage: stream_speed.py NAME [--nest models/g12-nest/model.nest] [--ram GB] [--reps 3]
       [--tiers 1.0,0.9,0.85,1.0:d,0.85:d] [--gen 128]
Writes results/speed/NAME.json
"""
import argparse
import json
import os
import statistics
import time

import mlx.core as mx

from nest.io import read_jsonl
from nest.runtime import NestedGemma

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--nest", default="models/g12-nest/model.nest")
ap.add_argument("--ram", type=float, default=None)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--tiers", default="1.0,0.9,0.85,1.0:d,0.85:d")
ap.add_argument("--prompt-tokens", type=int, default=512)
ap.add_argument("--gen", type=int, default=128)
args = ap.parse_args()

mx.set_cache_limit(1024**3)
M = NestedGemma.load(args.nest, ram_budget_gb=args.ram)
lm = M.model.language_model
prompt = mx.random.randint(1000, 200000, (1, args.prompt_tokens), key=mx.random.key(0))
texts = [r["user"] for r in read_jsonl("data/bench/gsm8k.jsonl")[:3]]
texts += [r["user"] for r in read_jsonl("data/heldout.jsonl") if r["source"] in ("chat", "code")][:3]
tiers = []
for t in args.tiers.split(","):
    w, _, d = t.partition(":")
    tiers.append((float(w), d == "d"))


def prefill():
    b0 = M.store.bytes
    t0 = time.perf_counter()
    h = lm.model(prompt, cache=lm.make_cache())
    mx.eval(mx.argmax(lm.logits_from_hidden(h[:, -1:]), axis=-1))
    dt = time.perf_counter() - t0
    return {"prefill_tps": prompt.shape[1] / dt, "time_to_answer_s": dt, "ssd_gb": (M.store.bytes - b0) / 1e9}


def decode():
    v = [M.generate(t, max_tokens=args.gen)["stats"]["tok_per_s"] for t in texts]
    return statistics.median(x for x in v if x)


def compute_only():
    """Same tier, but every streamed tensor is read into RAM first: no reads during the pass."""
    got = {n: M.store.read(n, r) for i in M.streamed_layers for n, r in M.plan[i]}
    M.prefetch.get = lambda i, wanted: {n: got[n] for n, _ in wanted}
    M.prefetch.start = lambda i, wanted: None
    try:
        return prefill()
    finally:
        del M.prefetch.get, M.prefetch.start, got


def reads_only():
    t = time.perf_counter()
    n = 0
    for i in M.streamed_layers:
        for name, rows in M.plan[i]:
            M.store.read(name, rows)
            n += M._bytes(name, rows)
    dt = time.perf_counter() - t
    return {"read_s": dt, "gb": n / 1e9, "gbps": n / dt / 1e9 if dt else None}


names = [M.tier_name(w, d) for w, d in tiers]
runs = {n: [] for n in names}
ctrl = {n: {"compute_only": [], "reads_only": []} for n in names}
switch = {n: [] for n in names}
for rep in range(args.reps + 1):            # round 0 is warm-up
    for (w, d), n in zip(tiers, names):
        ms = M.set(width=w, draft=d)
        r = prefill()
        r["decode_tps"] = decode()
        if rep == 0:
            continue
        switch[n].append(ms)
        runs[n].append(r)
        if M.streamed_layers:
            ctrl[n]["compute_only"].append(compute_only())
            ctrl[n]["reads_only"].append(reads_only())
        print(f"round {rep} {n}: prefill {r['prefill_tps']:.0f} tok/s, decode {r['decode_tps']:.2f} tok/s, "
              f"answer {r['time_to_answer_s']:.2f}s, ssd {r['ssd_gb']:.2f} GB, switch {ms:.2f} ms", flush=True)

out = {"ram_budget_gb": args.ram, "resident_gb": M.resident_bytes / 1e9, "load_s": M.load_s}
for n in names:
    row = {k: statistics.median(x[k] for x in runs[n]) for k in runs[n][0]}
    row["switch_ms_median"] = statistics.median(switch[n])
    for c in ("compute_only", "reads_only"):
        if ctrl[n][c]:
            row[c] = {k: statistics.median(x[k] for x in ctrl[n][c]) for k in ctrl[n][c][0]}
    out[n] = row
    print(n, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
out["_peak_gb"] = mx.get_peak_memory() / 1e9
os.makedirs("results/speed", exist_ok=True)
json.dump(out, open(f"results/speed/{args.name}.json", "w"), indent=1)
