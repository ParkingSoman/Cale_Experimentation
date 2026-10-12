"""Model objects against a running Cale (start one with scripts/cale_dev_server.py HOME).

usage: python cale/tests/objects.py [--base http://127.0.0.1:47899]
Nothing here belongs to one computer: sizes and speeds come from the machine under test.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cale.client import CaleError, ensure_running, model  # noqa: E402
from nest.levels3n import load_levels, pct_range  # noqa: E402

MIN_PCT, MAX_PCT = pct_range(load_levels())
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
a = ap.parse_args()
os.environ["CALE_PORT"] = str(urlparse(a.base).port)      # ensure_running() looks at this port
FAIL = []


def call(path, body=None, key=None):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(a.base + path, data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        r = urllib.request.urlopen(req, timeout=900)
        return r.status, dict(r.headers), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read())


def chat(mdl, key, prompt="Name the capital of France in one sentence.", **kw):
    return call("/v1/chat/completions", {"model": mdl, "messages": [{"role": "user", "content": prompt}], **kw}, key)


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def refused(**spec):
    """The error a bad model object gets, as (status, code)."""
    try:
        with model(a.base + "/v1", key, **spec):
            return None
    except CaleError as e:
        return e.status, json.loads(e.body)["error"]["code"]


def model_ids(k):
    return [m["id"] for m in call("/v1/models", key=k)[2]["data"]]


key = call("/cale/connect", {"app": "Objects Test"})[2]["api_key"]
other = call("/cale/connect", {"app": "Other Objects App"})[2]["api_key"]
base = a.base + "/v1"

check("ensure_running says Cale is already running", "already running" in ensure_running(5))

# 1. bad objects are refused with a clear code
check("nothing given: 400", refused() == (400, "invalid_model_object"), refused())
check("size with a deadline: 400", refused(size="fast", deadline_s=5, max_tokens=10) == (400, "invalid_model_object"))
check("deadline without max_tokens: 400", refused(deadline_s=5) == (400, "max_tokens_needed"))
s, _, r = call("/cale/models", {"size": "fast", "speed": 3}, key)
check("unknown field (raw request): 400", s == 400 and r["error"]["code"] == "invalid_model_object", r)
check("size 'auto': 400", refused(size="auto") == (400, "invalid_model_object"))

# 2. a size object: exact size, listed for its app only, used by name, gone when closed
with model(base, key, name="Small Tasks", size="fast", max_tokens=20) as m:
    check("id is made from the name", m.id == "small-tasks", m.id)
    with model(base, key, name="Small Tasks", size="fast") as m2:
        check("a second live object with the same name gets a suffix", m2.id.startswith("small-tasks-"), m2.id)
    with model(base, other, name="Small Tasks", size="fast") as m3:
        check("another app may use the same name", m3.id == "small-tasks", m3.id)
        s, hd, r = chat("small-tasks", key)
        check("...and each app reaches only its own", s == 200 and r["model"] == "small-tasks")
    check("size object holds the smallest size", m.info["size_pct"] == MIN_PCT, m.info)
    check("listed for its own app", m.id in model_ids(key))
    check("not listed for another app", m.id not in model_ids(other))
    s, hd, r = chat(m.id, key)
    check("chat by id runs the object's size", s == 200 and hd["X-Cale-Size"] == str(MIN_PCT), (s, hd.get("X-Cale-Size")))
    check("reply names the object", r["model"] == m.id, r.get("model"))
    check("object's answer length is the default", r["usage"]["completion_tokens"] <= 20, r["usage"])
    s, hd, r = chat(m.id, key, max_tokens=5)
    check("a call can still ask for fewer tokens", s == 200 and r["usage"]["completion_tokens"] <= 5, r["usage"])
    s, hd, r = chat(m.id, other)
    check("another app cannot use it: 404", s == 404 and r["error"]["code"] == "model_not_found", r)
    s, hd, r = chat(m.id, key, cale_deadline_s=5)
    check("cale_* fields with an object: 400", s == 400 and r["error"]["code"] == "goal_in_object", r)
check("object gone after the block", m.id not in model_ids(key))
s, hd, r = chat(m.id, key)
check("a closed object is not found: 404", s == 404, (s, r))

# 3. a deadline object: Cale picks the size per call. The deadline comes from this computer.
s, hd, r = chat("gemma-3n:auto", key, max_tokens=40, cale_deadline_s=3600, cale_allow_disk=True)
full_s = float(hd["X-Cale-Predicted-S"])
deadline = round(full_s * 1.5, 1)
with model(base, key, name="Answers", deadline_s=deadline, max_tokens=40, allow_disk=True) as m:
    check("deadline object holds no memory", not m.info.get("hold"), m.info)
    s, hd, r = chat(m.id, key)
    fast_pct = int(hd["X-Cale-Size"])
    check(f"normal speed: full size fits the deadline ({deadline} s)", s == 200 and fast_pct == MAX_PCT and hd["X-Cale-Goal-Met"] == "true", dict(hd))
    call("/cale/debug/slow", {"factor": 3.0})
    s, hd, r = chat(m.id, key)
    slow_pct = int(hd["X-Cale-Size"])
    call("/cale/debug/slow", {"factor": 1.0})
    check("3x slower computer: the same object picks a smaller size", slow_pct < fast_pct, (fast_pct, slow_pct))
with model(base, key, name="Impossible", deadline_s=0.2, max_tokens=40, on_miss="refuse") as m:
    s, hd, r = chat(m.id, key)
    check("impossible deadline + refuse: 422 with estimate", s == 422 and "predicted_s" in r["error"]["cale"], r)
with model(base, key, name="Impossible Run", deadline_s=0.2, max_tokens=40) as m:
    s, hd, r = chat(m.id, key)
    check("impossible deadline: runs smallest, says goal not met", s == 200 and hd["X-Cale-Goal-Met"] == "false"
          and int(hd["X-Cale-Size"]) == MIN_PCT, dict(hd))

# 4. a memory object holds what fits
with model(base, key, name="Tiny Memory", memory_pct=1) as m:
    check("memory 1%: smallest size, goal not met", m.info["size_pct"] == MIN_PCT and not m.info["goal_met"], m.info)
check("memory 1% + refuse: 422", refused(memory_pct=1, on_miss="refuse") == (422, "goal_missed"))
with model(base, key, name="Roomy", memory_pct=90, deadline_s=3600, max_tokens=10) as m:
    check("memory 90% + deadline: holds the full size and runs it", m.info["size_pct"] == MAX_PCT, m.info)
    s, hd, r = chat(m.id, key)
    check("...and the call uses it", s == 200 and hd["X-Cale-Size"] == str(MAX_PCT), dict(hd))

# 5. an app that quits loses its objects
import subprocess  # noqa: E402
code = (f"import sys; sys.path.insert(0, {str(ROOT)!r})\nfrom cale.client import model\nimport time\n"
        f"with model({base!r}, {other!r}, name='Doomed', size='fast') as m:\n    print(m.id, flush=True); time.sleep(600)\n")
p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
oid = p.stdout.readline().strip()
check("child's object is listed", oid in model_ids(other), oid)
p.kill()
t = time.time()
while time.time() - t < 8 and oid in model_ids(other):
    time.sleep(0.5)
check("killed app: its object is gone within 5 s", oid not in model_ids(other) and time.time() - t < 6, round(time.time() - t, 1))

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
