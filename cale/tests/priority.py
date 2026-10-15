"""An app that asked for speed is not stuck behind a background job. Against a running Cale (scripts/cale_dev_server.sh HOME).

usage: python cale/tests/priority.py [--base http://127.0.0.1:47899]
Setup: a background app (power "light", no speed goal) starts a long answer. A second background answer and an urgent one
(a first-word goal) arrive after it. The rule (cale/batching.py tier): the urgent answer is served before the second
background answer, and, on an engine that can run answers together, it does not wait for the first one either.
Nothing here belongs to one computer: the sizes and speeds come from the machine under test.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
ap.add_argument("--long", type=int, default=120, help="tokens of the background answers")
a = ap.parse_args()
os.environ["CALE_PORT"] = str(urlparse(a.base).port)
from cale.client import model  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def call(path, body=None, key=None):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(a.base + path, data=None if body is None else json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def stream(model_id, key, prompt, out, name):
    """Run one streamed answer; out[name] = {first, end} (seconds since the test started)."""
    body = {"model": model_id, "stream": True, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    rec = {"first": None, "end": None, "tokens": 0}
    out[name] = rec
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            ev = json.loads(line[5:])
            if ev.get("cale"):
                rec["cale"] = ev["cale"]
            if any((c.get("delta") or {}).get("content") for c in ev.get("choices", [])):
                rec["tokens"] += 1
                if rec["first"] is None:
                    rec["first"] = time.time() - T0
    rec["end"] = time.time() - T0


bg_key = call("/cale/connect", {"app": "Priority Test Background"})["api_key"]
fg_key = call("/cale/connect", {"app": "Priority Test Foreground"})["api_key"]
base = a.base + "/v1"
call("/v1/chat/completions", {"model": "gemma-3n:fast", "max_tokens": 2, "messages": [{"role": "user", "content": "hi"}]}, bg_key)   # the model is loaded

with model(base, bg_key, name="background", power="light", max_tokens=a.long) as bg, \
        model(base, bg_key, name="background2", power="light", max_tokens=a.long) as bg2, \
        model(base, fg_key, name="urgent", max_first_token_s=30, max_tokens=12) as fg:
    out = {}
    T0 = time.time()
    t1 = threading.Thread(target=stream, args=(bg.id, bg_key, "Count slowly from 1 to 100, one number per line.", out, "bg1"))
    t1.start()
    while "bg1" not in out or out["bg1"]["first"] is None:
        time.sleep(0.05)
    t2 = threading.Thread(target=stream, args=(bg2.id, bg_key, "Count slowly from 1 to 100, one number per line.", out, "bg2"))
    t2.start()
    time.sleep(0.3)
    t3 = threading.Thread(target=stream, args=(fg.id, fg_key, "Say hello in five words.", out, "urgent"))
    t3.start()
    for t in (t1, t2, t3):
        t.join()

for k in ("bg1", "bg2", "urgent"):
    r = out[k]
    print(f"   {k:7s} first word at {r['first']:.2f} s, finished at {r['end']:.2f} s, {r['tokens']} pieces")
check("every answer finished", all(out[k]["end"] for k in out))
check("the urgent answer finished before the second background answer", out["urgent"]["end"] < out["bg2"]["end"],
      f"urgent {out['urgent']['end']:.2f} s, background 2 {out['bg2']['end']:.2f} s")
check("the urgent answer did not wait for the second background answer to start",
      out["urgent"]["first"] < out["bg2"]["first"], f"{out['urgent']['first']:.2f} vs {out['bg2']['first']:.2f}")
st = call("/cale/status")
together = bool(st.get("slots") and st["slots"] > 1)
if together:
    check("answers run together here: the urgent answer did not wait for the first background answer",
          out["urgent"]["end"] < out["bg1"]["end"], f"urgent {out['urgent']['end']:.2f} s, background 1 {out['bg1']['end']:.2f} s")
else:
    print("   (this engine answers one at a time: the running answer finishes first, the rule only picks who starts next)")

# Time an answer spends paused for a higher tier is waiting, not working: the reply says so, and the predictor does not learn it as slow.
paused = {k: round(float((out[k].get("cale") or {}).get("paused_pre_s") or 0) + float((out[k].get("cale") or {}).get("paused_dec_s") or 0), 2) for k in out}
print(f"   paused seconds: {paused}")
check("the reply reports paused seconds", all("paused_pre_s" in (out[k].get("cale") or {}) for k in out), {k: list((out[k].get("cale") or {})) for k in out})
check("the urgent answer was never paused", paused["urgent"] == 0.0, paused)
if together:
    check("a background answer was paused while the urgent one ran", max(paused["bg1"], paused["bg2"]) > 0.0, paused)
    wall = float((out["bg1"].get("cale") or {}).get("seconds") or 0)
    check("paused seconds are never more than the answer's own time", all(paused[k] <= float((out[k].get("cale") or {}).get("seconds") or 0) + 0.01 for k in out), paused)

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
