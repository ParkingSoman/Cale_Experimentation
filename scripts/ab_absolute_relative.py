#!/usr/bin/env python3
"""Absolute vs relative: the same notes-style completion code on one computer, three ways of asking.

  absolute-smart  model "gemma-3n:smart"  (a developer who wants the best model)
  absolute-fast   model "gemma-3n:fast"   (a developer who tuned on a slow computer)
  relative        a model object with goals: first word in 1 s, answers at least 8 tokens/s

Every call is scored against ONE feel target, whatever the arm: first word within 1 s AND at least 8 tokens/s.
Size is the intelligence slice that ran (100 = full model). Against a running Cale.

usage: python3 scripts/ab_absolute_relative.py --base http://127.0.0.1:47899 --label mac [--rounds 3] [--out FILE]
"""
import argparse
import json
import statistics
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
ap.add_argument("--label", default="run")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--first-s", type=float, default=1.0)       # the feel target (also the relative arm's goals)
ap.add_argument("--tps", type=float, default=8.0)
ap.add_argument("--out", default=None)
a = ap.parse_args()

SYSTEM = ("You continue the user's text. Reply with ONLY the next few words that naturally follow, "
          "in the same language and style. Do not repeat the text. No quotes, no explanation.")
PROMPTS = ["Dear Maria, I wanted to follow up on", "The quarterly numbers look good, but we should",
           "Ingredients for the cake: two cups of", "Meeting notes: the team agreed that the next step is to",
           "To reset your password, open the settings page and", "The best part of the trip was when we"]


def call(path, body, key=None, timeout=900):
    h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}
    req = urllib.request.Request(a.base + path, data=json.dumps(body).encode(), headers=h)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


KEYS = {arm: call("/cale/connect", {"app": "AB " + arm})["api_key"] for arm in ("absolute-smart", "absolute-fast", "relative")}   # one app per arm: Cale never reuses a prompt across apps, so no arm helps another
rel = call("/cale/models", {"name": "ab-notes", "max_first_token_s": a.first_s, "min_tokens_per_s": a.tps,
                            "max_tokens": 20, "lease_s": 1800}, KEYS["relative"])["id"]
ARMS = {"absolute-smart": "gemma-3n:smart", "absolute-fast": "gemma-3n:fast", "relative": rel}


def one(arm, model, text):
    body = {"model": model, "stream": True, "max_tokens": 20, "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]}
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEYS[arm]})
    t0, first, n, info = time.time(), None, 0, {}
    with urllib.request.urlopen(req, timeout=900) as r:
        size = r.headers.get("X-Cale-Size")
        said = {"predicted_tps": r.headers.get("X-Cale-Predicted-Tokens-Per-S"), "predicted_first_s": r.headers.get("X-Cale-Predicted-First-Token-S"),
                "confidence": r.headers.get("X-Cale-Confidence"), "goal_met": r.headers.get("X-Cale-Goal-Met")}
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data:") and not line.endswith("[DONE]"):
                ev = json.loads(line[5:])
                if ev.get("cale"):
                    info = ev["cale"]
                if any((c.get("delta") or {}).get("content") for c in ev.get("choices", [])):
                    n += 1
                    if first is None:
                        first = time.time()
    end = time.time()
    tps = (n - 1) / (end - first) if first and n >= 4 and end > first else None
    return {"first_s": round(first - t0, 3) if first else None, "total_s": round(end - t0, 3), "tokens": n,
            "tps": round(tps, 2) if tps else None, "said": said, "size": float(info.get("size_pct") or size or 0) or None}


for arm, m in ARMS.items():                                   # warm up: load, first calls of every size
    one(arm, m, "Hello")
rows = {k: [] for k in ARMS}
for rnd in range(a.rounds):
    for p in PROMPTS:
        for arm, model in ARMS.items():                   # interleaved, so heat and load hit every arm alike
            rows[arm].append(one(arm, model, p + " " * rnd))


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


out = {"label": a.label, "when": time.strftime("%Y-%m-%d %H:%M"), "target": {"first_s": a.first_s, "tps": a.tps}, "arms": {}}
for arm, rs in rows.items():
    ok = [r for r in rs if r["first_s"] is not None and r["first_s"] <= a.first_s and r["tps"] is not None and r["tps"] >= a.tps]
    out["arms"][arm] = {"calls": len(rs), "feel_met": len(ok), "first_s_median": med(r["first_s"] for r in rs),
                        "first_s_max": max((r["first_s"] for r in rs if r["first_s"] is not None), default=None),
                        "tps_median": med(r["tps"] for r in rs), "total_s_median": med(r["total_s"] for r in rs),
                        "size_median": med(r["size"] for r in rs), "rows": rs}
print(f"{a.label}: target first word <= {a.first_s} s and >= {a.tps} tok/s")
print(f"{'arm':16} {'feel met':>9} {'first word':>11} {'tok/s':>7} {'whole call':>11} {'size %':>7}")
for arm, v in out["arms"].items():
    print(f"{arm:16} {v['feel_met']:>4}/{v['calls']:<4} {v['first_s_median']!s:>10}s {v['tps_median']!s:>7} {v['total_s_median']!s:>10}s {v['size_median']!s:>7}")
if a.out:
    open(a.out, "w").write(json.dumps(out, indent=1))
    print("saved", a.out)
