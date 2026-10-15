#!/usr/bin/env python3
"""Answer speed under controlled load: how many answers at once (crowd) and how much other programs use the CPU (external).
Against a running Cale. Standard library only. Fixed sizes (no size changes), temperature 0, short answers.

usage: python3 scripts/load_sweep.py --base http://127.0.0.1:47899 [--sizes 100,48] [--crowd 1,2,3] [--external 0,0.5,1.0] [--tokens 40] [--out FILE]
Each cell: `crowd` threads each ask 2 answers in a row. External load = busy processes (share of the cores). Prints speed per cell and the slowdown vs the
quiet single answer of the same size. That slowdown is what the predictor's load model must learn (cale/predict.py live_factor assumes at most 0.5x).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
ap.add_argument("--sizes", default="100,48")
ap.add_argument("--crowd", default="1,2,3")
ap.add_argument("--external", default="0,0.5,1.0")
ap.add_argument("--tokens", type=int, default=40)
ap.add_argument("--out", default=None)
a = ap.parse_args()


def call(path, body, key=None, timeout=900):
    h = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}
    req = urllib.request.Request(a.base + path, data=json.dumps(body).encode(), headers=h)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


key = call("/cale/connect", {"app": "Load Sweep"})["api_key"]
PROMPTS = ["Write a short story about a lighthouse keeper.", "Explain how a bicycle stays upright, in simple words.", "List ten facts about the ocean."]


def one(model, i, out):
    body = {"model": model, "stream": True, "max_tokens": a.tokens, "temperature": 0,
            "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]}
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0, first, n = time.time(), None, 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data:") and not line.endswith("[DONE]"):
                ev = json.loads(line[5:])
                if any((c.get("delta") or {}).get("content") for c in ev.get("choices", [])):
                    n += 1
                    if first is None:
                        first = time.time()
    end = time.time()
    if first and n >= 9 and end > first:
        out.append((n - 1) / (end - first))


def hogs(share):
    n = int(round(share * (os.cpu_count() or 1)))
    return [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(n)]


call("/v1/chat/completions", {"model": "gemma-3n:100", "max_tokens": 2, "messages": [{"role": "user", "content": "hi"}]}, key)
rows = []
for size in a.sizes.split(","):
    model = f"gemma-3n:{size}"
    one(model, 0, [])                                        # settle the size before measuring
    base = None
    for ext in [float(x) for x in a.external.split(",")]:
        for crowd in [int(x) for x in a.crowd.split(",")]:
            procs = hogs(ext)
            time.sleep(1.0 if procs else 0)
            out = []
            ts = [threading.Thread(target=lambda k=k: [one(model, k + j, out) for j in range(2)]) for k in range(crowd)]
            t = time.time()
            [x.start() for x in ts]
            [x.join() for x in ts]
            for p in procs:
                p.kill()
            tps = statistics.median(out) if out else None
            if ext == 0 and crowd == 1:
                base = tps
            row = {"size": int(size), "crowd": crowd, "external": ext, "calls": len(out), "tps_each": tps,
                   "tps_total": (tps * crowd) if tps else None, "vs_quiet_single": (tps / base) if tps and base else None, "seconds": round(time.time() - t, 1)}
            rows.append(row)
            print(f"size {size:>3s}%  crowd {crowd}  external {ext:.1f}  each {tps and round(tps, 2)} tok/s  total {row['tps_total'] and round(row['tps_total'], 2)}  "
                  f"vs quiet single {row['vs_quiet_single'] and round(row['vs_quiet_single'], 2)}", flush=True)
            time.sleep(1)
if a.out:
    open(a.out, "w").write(json.dumps(rows, indent=1))
