#!/usr/bin/env python3
"""Does the speed-goal adjuster change sizes for good reasons? App A has a speed goal and answers again and again. App B reads long prompts
in the background (prompt reading steals steps from A, which is the case the old rule mistook for a slow size).
Against a running Cale. Prints A's number of size changes, mean size, median speed and the share of answers that met the goal.

usage: python3 scripts/adjust_ab.py --base http://127.0.0.1:47899 --goal 5 [--seconds 90] [--label old] [--out FILE]
Run it twice, once with CALE_ADJUST_MODE=old and once with sustained, and compare.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
ap.add_argument("--goal", type=float, default=5.0)
ap.add_argument("--seconds", type=float, default=90)
ap.add_argument("--label", default="run")
ap.add_argument("--out", default=None)
a = ap.parse_args()


def call(path, body, key=None, timeout=900):
    h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}
    req = urllib.request.Request(a.base + path, data=json.dumps(body).encode(), headers=h)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


ka = call("/cale/connect", {"app": "AB Speed App"})["api_key"]
kb = call("/cale/connect", {"app": "AB Reader App"})["api_key"]
ma = call("/cale/models", {"name": "ab-fast", "min_tokens_per_s": a.goal, "lease_s": 900}, ka)["id"]
mb = call("/cale/models", {"name": "ab-bg", "power": "light", "lease_s": 900}, kb)["id"]
call("/v1/chat/completions", {"model": ma, "max_tokens": 2, "messages": [{"role": "user", "content": "hi"}]}, ka)
stop = threading.Event()
reads = [0]


def reader():
    n = 0
    while not stop.is_set():
        n += 1
        text = f"Document {n}. " + " ".join(f"Sentence {i} of the filler text about harbors and ships number {n}." for i in range(70))
        try:
            call("/v1/chat/completions", {"model": mb, "max_tokens": 6, "messages": [{"role": "user", "content": text + "\nSay OK."}]}, kb)
            reads[0] += 1
        except Exception:
            time.sleep(0.5)


def answer(i):
    body = {"model": ma, "stream": True, "max_tokens": 60, "temperature": 0,
            "messages": [{"role": "user", "content": f"Write a short story about a lighthouse keeper, version {i}."}]}
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + ka})
    first, n, info = None, 0, {}
    with urllib.request.urlopen(req, timeout=900) as r:
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
    return ((n - 1) / (end - first) if first and n >= 9 else None), info


t = threading.Thread(target=reader, daemon=True)
t.start()
rows, i, t0 = [], 0, time.time()
while time.time() - t0 < a.seconds:
    i += 1
    tps, info = answer(i)
    head = (tps / a.goal) if tps else None
    rows.append({"tps": tps, "size": info.get("size"), "changes": len(info.get("size_changes") or []), "met": info.get("goal_met_actual"),
                 "headroom": head, "wasted": bool(head and head > 1.3 and (info.get("size") or 100) < 99.5)})
stop.set()
t.join(timeout=60)
tps = [r["tps"] for r in rows if r["tps"]]
out = {"label": a.label, "goal": a.goal, "answers": len(rows), "reads_by_B": reads[0], "size_changes": sum(r["changes"] for r in rows),
       "mean_size_pct": round(statistics.mean(r["size"] for r in rows if r["size"]), 1) if any(r["size"] for r in rows) else None,
       "median_tps": round(statistics.median(tps), 2) if tps else None,
       "met_share": round(sum(1 for r in rows if r["met"]) / len(rows), 2) if rows else None,
       "headroom_median": round(statistics.median(r["headroom"] for r in rows if r["headroom"]), 2) if any(r["headroom"] for r in rows) else None,
       "wasted_share": round(sum(1 for r in rows if r["wasted"]) / len(rows), 2) if rows else None,
       "sizes_seen": sorted({r["size"] for r in rows if r["size"]})}
print(json.dumps(out), flush=True)
if a.out:
    open(a.out, "w").write(json.dumps({**out, "rows": rows}, indent=1))
