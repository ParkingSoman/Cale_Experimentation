"""Step 6b: speed per rung of the nested file, round-robin, medians.

Per tier: prefill tok/s (512-token prompt), decode tok/s, time-to-answer for a one-pass
call, SSD bytes per pass. For a tier that streams, two controls:
  compute-only  same math, but every layer reuses layer 0's streamed blocks (no reads)
  reads-only    read the streamed bytes of one pass, no compute

usage: stream_speed.py NEST NAME [--ram 12] [--reps 5] [--tiers 4:1.0,4:0.85,8:1.0,16:1.0]
       [--gen 128] [--gen16 8]
Writes results/speed/NAME.json
"""
import argparse
import json
import os
import statistics
import time

import mlx.core as mx

from nest.runtime import NestedGemma

ap = argparse.ArgumentParser()
ap.add_argument("nest")
ap.add_argument("name")
ap.add_argument("--ram", type=float, default=12.0)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--tiers", default="4:1.0,4:0.85,8:1.0,16:1.0")
ap.add_argument("--prompt-tokens", type=int, default=512)
ap.add_argument("--gen", type=int, default=128)
ap.add_argument("--gen16", type=int, default=8, help="decode tokens for tiers that stream")
args = ap.parse_args()

mx.set_cache_limit(1024**3)
M = NestedGemma.load(args.nest, ram_budget_gb=args.ram)
lm = M.model.language_model
prompt = mx.random.randint(1000, 200000, (1, args.prompt_tokens), key=mx.random.key(0))
tiers = [(int(b), float(w)) for b, w in (t.split(":") for t in args.tiers.split(","))]


def one_run(n_gen):
    b0 = M.store.bytes
    cache = lm.make_cache()
    t0 = time.perf_counter()
    h = lm.model(prompt, cache=cache)
    tok = mx.argmax(lm.logits_from_hidden(h[:, -1:]), axis=-1)
    mx.eval(tok)
    t1 = time.perf_counter()
    for _ in range(n_gen):
        h = lm.model(tok.reshape(1, 1), cache=cache)
        tok = mx.argmax(lm.logits_from_hidden(h[:, -1:]), axis=-1)
        mx.async_eval(tok)
    mx.eval(tok)
    t2 = time.perf_counter()
    return {"prefill_tps": prompt.shape[1] / (t1 - t0), "time_to_answer_s": t1 - t0,
            "decode_tps": n_gen / (t2 - t1), "ssd_gb": (M.store.bytes - b0) / 1e9}


def compute_only(n_gen):
    """Same tier, but streamed tensors come from RAM: layer 0's blocks reused by every layer."""
    plan = M.plan
    first = M.streamed_layers[0]
    got = {n: M.store.read(n, r) for n, r in plan[first]}
    alias = {}
    for i in M.streamed_layers:
        for (n, r), (n0, _) in zip(plan[i], plan[first]):
            alias[n] = got[n0]
    saved = M.prefetch.get

    M.prefetch.get = lambda i, wanted: {n: alias[n] for n, _ in wanted}
    M.prefetch.start = lambda i, wanted: None
    try:
        return one_run(n_gen)
    finally:
        M.prefetch.get = saved
        del M.prefetch.start


def reads_only():
    t = time.perf_counter()
    n = 0
    for i in M.streamed_layers:
        for name, rows in M.plan[i]:
            M.store.read(name, rows)
            n += M._bytes(name, rows)
    dt = time.perf_counter() - t
    return {"read_s": dt, "gb": n / 1e9, "gbps": n / dt / 1e9}


names = [f"{b}bit-w{round(w * 100)}" for b, w in tiers]
runs = {n: [] for n in names}
ctrl = {n: {"compute_only": [], "reads_only": []} for n in names}
switch = {n: [] for n in names}
for rep in range(args.reps + 1):            # round 0 is warm-up
    for (b, w), n in zip(tiers, names):
        ms = M.set(b, w)
        g = args.gen if not M.streamed_layers else args.gen16
        r = one_run(g)
        if rep == 0:
            continue
        switch[n].append(ms)
        runs[n].append(r)
        if M.streamed_layers:
            ctrl[n]["compute_only"].append(compute_only(g))
            ctrl[n]["reads_only"].append(reads_only())
        print(f"round {rep} {n}: prefill {r['prefill_tps']:.0f} tok/s, decode {r['decode_tps']:.2f} tok/s, "
              f"answer {r['time_to_answer_s']:.2f}s, ssd {r['ssd_gb']:.1f} GB", flush=True)

out = {"ram_budget_gb": args.ram, "resident_gb": M.resident_bytes / 1e9, "load_s": M.load_s}
for n in names:
    med = lambda key, xs=runs[n]: statistics.median(x[key] for x in xs)
    row = {k: med(k) for k in runs[n][0]}
    row["switch_ms_median"] = statistics.median(switch[n])
    for c in ("compute_only", "reads_only"):
        if ctrl[n][c]:
            row[c] = {k: statistics.median(x[k] for x in ctrl[n][c]) for k in ctrl[n][c][0]}
    out[n] = row
    print(n, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
out["_peak_gb"] = mx.get_peak_memory() / 1e9
os.makedirs("results/speed", exist_ok=True)
json.dump(out, open(f"results/speed/{args.name}.json", "w"), indent=1)
