"""Score table for the whole-model distilled students (plan 2026-11-25 "distill the whole of A").

Reads only saved .json files in neuron/results/coarse/r18/distill_whole/ (no GPU, no photos).
One scale for all sizes: test KL to A (5,000 test photos) and test top-1.
Winning start per size = lower HELD KL (seed 0, 1x length). A 2x run replaces its 1x run only if its held KL is lower
by more than 1% (ladder tie rule).
Usage: python distill/score.py
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "neuron", "results", "coarse", "r18", "distill_whole")

runs = {}
for f in sorted(glob.glob(os.path.join(D, "W*_s*.json"))):
    m = re.match(r"W(\d+)([RP])_s(\d)(_x2)?\.json$", os.path.basename(f))
    if m:
        runs[(int(m[1]), m[2], int(m[3]), bool(m[4]))] = json.load(open(f))

print("| Neurons kept | Start | Seed | Weights | lr | Stop / steps | Held KL before training | Held KL | Test KL | Test top-1 | Minutes |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for (p, st, s, x2), r in sorted(runs.items(), key=lambda kv: (-kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
    print(f"| {p}% | {st} | {s}{' (2x)' if x2 else ''} | {r['weights']:,} | {r['lr']:g} | {r['stop_step']} / {r['steps']} "
          f"| {r['curve'][0]['held_kl']:.4f} | {r['held_kl']:.4f} | {r['test']['kl']:.4f} | {100 * r['test']['top1']:.1f}% | {r['minutes']} |")
if runs:
    print(f"\nA on the same test photos: top-1 {100 * next(iter(runs.values()))['A_test']['top1']:.1f}%, KL 0\n")

for p in sorted({k[0] for k in runs}, reverse=True):
    cand = {st: runs.get((p, st, 0, False)) for st in "RP"}
    cand = {st: r for st, r in cand.items() if r}
    if not cand:
        continue
    win = min(cand, key=lambda st: cand[st]["held_kl"])
    line = f"{p}%: winning start {win} (held KL " + ", ".join(f"{st} {r['held_kl']:.4f}" for st, r in cand.items()) + ")"
    r1, r2 = runs.get((p, win, 0, False)), runs.get((p, win, 0, True))
    if r2:
        use = "2x" if r2["held_kl"] < 0.99 * r1["held_kl"] else "1x"
        line += f"; 2x held {r2['held_kl']:.4f} vs 1x {r1['held_kl']:.4f} -> use {use}"
    s1 = runs.get((p, win, 1, False))
    if s1:
        line += f"; seed gap in test KL {abs(s1['test']['kl'] - r1['test']['kl']):.4f}"
    print(line)

# Wave 3 (plan 2026-11-26). Win rule: best RAW held KL lower than the baseline's by more than 2% (the photo-order gap
# at 50% P is 1.6%, the two-seed gap at 5% R is 0.8%). Wave 1-2 runs had raw weights only, so the EMA copy is its own
# column and never decides a win. Baselines: 50% = W50P_s0 (wave 1). 5% = R in the same --multi process (same batches,
# same code): W5R_s0_m for the _m variants, W5R_s0_m0 for the _m0 variants.
def raw_best(r):
    return min(c["held_kl"] for c in r["curve"])


def ema_best(r):
    v = [c["held_kl_ema"] for c in r["curve"] if "held_kl_ema" in c]
    return min(v) if v else float("nan")


w3 = {}
for f in sorted(glob.glob(os.path.join(D, "W*_s*.json"))):
    t = os.path.basename(f)[:-5]
    if re.match(r"W(\d+)([RP])_s(\d)(_x2)?$", t) or "smoke" in t or "_ctl" in t:
        continue
    w3[t] = json.load(open(f))


def base_of(t, r):
    p = round(r["f"] * 100)
    if p == 5 and t.endswith(("_m", "_m0")):
        b = "W5R_s0" + t[t.rindex("_"):]
        return (b, w3[b]) if b in w3 and b != t else (None, None)
    if p == 50:
        return "W50P_s0", runs.get((50, "P", 0, False))
    return None, None


if w3:
    print("\nWave 3\n")
    print("| Tag | Start | lr | wd | clip | mix | Extra photos | Best raw held KL | vs baseline | Best EMA held KL "
          "| Kept (step, from) | Test KL | Test top-1 | Minutes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t, r in sorted(w3.items(), key=lambda kv: (-round(kv[1]["f"] * 100), kv[0])):
        bt, b = base_of(t, r)
        rb = raw_best(r)
        vs = ""
        if b:
            bk = raw_best(b) if "curve" in b else b["held_kl"]
            vs = f"{100 * (rb / bk - 1):+.1f}% vs {bt}" + (" WIN" if rb < 0.98 * bk else "")
        print(f"| {t} | {r['start']} | {r['lr']:g} | {r.get('wd', '')} | {r.get('clip', '')} | {r.get('mix', '')} "
              f"| {r.get('extra_photos', 0):,} | {rb:.4f} | {vs} | {ema_best(r):.4f} "
              f"| {r['stop_step']} / {r['steps']} ({r.get('best_from', 'raw')}) | {r['test']['kl']:.4f} "
              f"| {100 * r['test']['top1']:.1f}% | {r.get('minutes', '')} |")
