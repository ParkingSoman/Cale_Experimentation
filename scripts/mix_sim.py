"""Step 6c: mixed workload. Total time and quality of 100 calls, from all-easy to all-hard,
for each way of serving them. Uses only measured numbers.

| Call type | Stand-in task (quality) | Prompt / output tokens |
|---|---|---|
| easy, long output   | IFEval   | 200 / 200 |
| hard, long output   | GSM8K    | 150 / 250 |
| hard, one pass      | MMLU-Pro | 300 / 0   |

Hard calls are split half long-output, half one-pass. Routing is perfect by construction
(the caller knows the call type). Time per call = prompt / prefill tok/s + output / decode tok/s,
plus a reload when a one-at-a-time stack changes model. Long-output calls use the
effective speed measured on matching content (results/speed/content.json: generated
tokens / whole-call seconds), because the drafter's speed depends on the text.
One-pass calls use prompt / prefill tok/s.

Check: the same formula is applied to the calls of the direct run (results/mix-direct.json).

usage: mix_sim.py   (reads results/speed/{nest,e4b-now,content}.json, results/stack-cost.json,
                     results/mix-direct.json, results/bench/*.json; writes results/mix-sim.json)
"""
import json
import random

SP = json.load(open("results/speed/nest.json"))
E4B_SP = json.load(open("results/speed/e4b-now.json"))["full"]
STACK = json.load(open("results/stack-cost.json"))
CONTENT = json.load(open("results/speed/content.json"))


def acc(name, level, task, n=None):
    items = json.load(open(f"results/bench/{name}.json"))[level][task]["items"][:n]
    return sum(i["ok"] for i in items) / len(items)


# Quality per tier and call type (same items for every tier: first 50 GSM8K, first 30 IFEval).
Q = {
    "w100": {"easy": acc("qg4-gen", "full", "ifeval"), "hard_long": acc("qg4-gen", "full", "gsm8k", 50),
             "hard_one": acc("nest-L", "w100", "mmlupro1k")},
    "w100+draft": {"easy": acc("nest-gen", "w100+draft", "ifeval"), "hard_long": acc("nest-gen", "w100+draft", "gsm8k"),
                   "hard_one": acc("nest-L", "w100", "mmlupro1k")},
    "w85+draft": {"easy": acc("nest-gen", "w85+draft", "ifeval"), "hard_long": acc("nest-gen", "w85+draft", "gsm8k"),
                  "hard_one": acc("qg4-L", "u85", "mmlupro1k")},
    "E4B": {"easy": acc("e4b-gen", "full", "ifeval"), "hard_long": acc("e4b-L", "full", "gsm8k", 50),
            "hard_one": acc("e4b-L", "full", "mmlupro1k")},
}
TOK = {"easy": (200, 200), "hard_long": (150, 250), "hard_one": (300, 0)}


def speed(tier):
    s = E4B_SP if tier == "E4B" else SP[tier]
    return s["prefill_tps"], s["decode_tps"]


def call_time(tier, kind, prompt=None, out=None):
    p, o = TOK[kind]
    p, o = (p if prompt is None else prompt), (o if out is None else out)
    if kind == "hard_one":
        return p / speed(tier)[0]
    return o / CONTENT[tier][kind]["effective_tps"]


STRATEGIES = {
    "always 12B w100": (lambda k: "w100", "nest"),
    "always 12B w100+draft": (lambda k: "w100+draft" if k != "hard_one" else "w100", "nest"),
    "always 12B w85+draft": (lambda k: "w85+draft", "nest"),
    "nest, developer-routed": (lambda k: {"easy": "w85+draft", "hard_long": "w100+draft", "hard_one": "w100"}[k], "nest"),
    "E4B only": (lambda k: "E4B", "e4b"),
    "12B+E4B stack, both in RAM": (lambda k: "E4B" if k == "easy" else ("w100+draft" if k == "hard_long" else "w100"), "both"),
    "12B+E4B stack, one at a time": (lambda k: "E4B" if k == "easy" else ("w100+draft" if k == "hard_long" else "w100"), "reload"),
}
RAM = {"nest": 7.8, "e4b": 5.2, "both": 7.8 + 5.2, "reload": 7.8}


def simulate(easy_frac, n=100, seed=0):
    rng = random.Random(seed)
    kinds = (["easy"] * round(n * easy_frac) + ["hard_long"] * round(n * (1 - easy_frac) / 2))
    kinds += ["hard_one"] * (n - len(kinds))
    rng.shuffle(kinds)
    out = {}
    for name, (route, mem) in STRATEGIES.items():
        t, q, cur, reloads = 0.0, 0.0, None, 0
        for k in kinds:
            tier = route(k)
            model = "E4B" if tier == "E4B" else "12B"
            if mem == "reload" and cur is not None and model != cur:
                t += STACK["E4B 4-bit (mlx-vlm)" if model == "E4B" else "12B qg4 (mlx-vlm)"]["warm_load_s"]
                reloads += 1
            cur = model
            t += call_time(tier, k)
            q += Q[tier][k]
        out[name] = {"total_s": round(t, 1), "quality": round(100 * q / n, 1), "ram_gb": RAM[mem], "reloads": reloads}
    return out


res = {f"{int(e * 100)}% easy": simulate(e) for e in (1.0, 0.8, 0.5, 0.2, 0.0)}
d = json.load(open("results/mix-direct.json"))
pred = sum(call_time(c["tier"], c["kind"], c["prompt_tokens"], c["tokens"]) for c in d["calls"])
check = {"measured_s": d["measured_total_s"], "predicted_s": pred, "error": (pred - d["measured_total_s"]) / d["measured_total_s"]}
print(f"direct-run check: measured {check['measured_s']:.1f}s, predicted {pred:.1f}s, error {100 * check['error']:+.1f}%")
json.dump({"quality": Q, "results": res, "direct_check": check}, open("results/mix-sim.json", "w"), indent=1)
for mix, rows in res.items():
    print(f"\n{mix} (100 calls)\n| strategy | total time (s) | quality | RAM GB | reloads |\n|---|---|---|---|---|")
    for name, r in rows.items():
        print(f"| {name} | {r['total_s']} | {r['quality']} | {r['ram_gb']} | {r['reloads']} |")
