"""Holds and relative goals against a running Cale (start one with scripts/cale_dev_server.py HOME).
No number here belongs to one computer: sizes, speeds and memory are read from the machine under test.

usage: python cale/tests/relative.py [--base http://127.0.0.1:47899]
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cale import oslayer  # noqa: E402
from cale.client import CaleError, hold  # noqa: E402
from cale.profile import WORK_BYTES  # noqa: E402
from nest.levels3n import load_levels, pct_range  # noqa: E402

MIN_PCT, MAX_PCT = pct_range(load_levels())
RAM_TOL = 0.15      # GB: process memory moves a little between readings
OVER_TOL = 0.30     # GB: Cale may predict a little more memory than is used. It must never predict less.


def predicted_ok(actual, predicted):
    return predicted >= actual - 0.05 and predicted - actual <= OVER_TOL

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
a = ap.parse_args()
FAIL = []


def call(path, body=None, key=None, raw=False):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(a.base + path, data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        r = urllib.request.urlopen(req, timeout=900)
        return (r.status, dict(r.headers), json.loads(r.read()))
    except urllib.error.HTTPError as e:
        return (e.code, dict(e.headers), json.loads(e.read()))


def chat(model, key, prompt="Name the capital of France in one sentence.", **kw):
    return call("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}], **kw}, key)


def status():
    return call("/cale/status")[2]


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok:
        FAIL.append(name)


def wait_for(pred, secs=15):
    t = time.time()
    while time.time() - t < secs:
        s = status()
        if pred(s):
            return s
        time.sleep(0.5)
    return status()


key = call("/cale/connect", {"app": "Relative Test"})[2]["api_key"]
other = call("/cale/connect", {"app": "Other App"})[2]["api_key"]

# 0. what the full size holds on this computer
with hold(a.base + "/v1", key, size="smart") as h0:
    full_gb = h0.info["ram_gb"]
print(f"full size holds {full_gb} GB here", flush=True)

# 1. hold the smallest size: memory is lower than the full model, and stays there
with hold(a.base + "/v1", key, size="fast") as h:
    check("hold fast: RAM below full", h.info["ram_gb"] < full_gb - 0.5, h.info)
    check("hold listed in status", len(status()["holds"]) == 1)
    base_gb = status()["resident_gb"]

    # 2. a smaller or equal size runs with no disk read
    s, hd, r = chat("gemma-3n:fast", key, max_tokens=24)
    check("held size: no disk read", s == 200 and hd["X-Cale-Disk-MB"] == "0", hd.get("X-Cale-Disk-MB"))

    # 3. a bigger size works, reads from disk, and memory falls back after
    s, hd, r = chat("gemma-3n:smart", key, max_tokens=24)
    check("bigger size answers", s == 200 and hd["X-Cale-Size"] == "100", (s, hd.get("X-Cale-Size")))
    check("bigger size read from disk", int(hd["X-Cale-Disk-MB"]) > 500, hd.get("X-Cale-Disk-MB"))
    check("memory back at the hold level", abs(status()["resident_gb"] - base_gb) < RAM_TOL, (status()["resident_gb"], base_gb))

    # 4. auto without deadline: largest size that needs no disk
    s, hd, r = chat("gemma-3n:auto", key, max_tokens=24)
    check("auto, no deadline: runs from RAM", s == 200 and hd["X-Cale-Disk-MB"] == "0" and int(hd["X-Cale-Size"]) <= MIN_PCT + 2, hd.get("X-Cale-Size"))

    # 5. auto with deadline and no disk permission stays at the held size; with permission it may go bigger
    s, hd, r = chat("gemma-3n:auto", key, max_tokens=24, cale_deadline_s=600)
    check("deadline, no disk allowed: held size", s == 200 and int(hd["X-Cale-Size"]) <= MIN_PCT + 2, hd.get("X-Cale-Size"))
    s, hd, r = chat("gemma-3n:auto", key, max_tokens=24, cale_deadline_s=600, cale_allow_disk=True)
    check("deadline, disk allowed: full size", s == 200 and hd["X-Cale-Size"] == "100" and hd["X-Cale-Goal-Met"] == "true", dict(hd))

    # 6. errors
    s, hd, r = chat("gemma-3n:auto", key, cale_deadline_s=5)
    check("deadline without max_tokens: 400", s == 400 and r["error"]["code"] == "max_tokens_needed", r)
    s, hd, r = chat("gemma-3n:smart", key, cale_deadline_s=5)
    check("goal field without auto: 400", s == 400 and r["error"]["code"] == "goal_needs_auto", r)
    s, hd, r = chat("gemma-3n:auto", key, max_tokens=24, cale_deadline_s=0.2)
    check("impossible deadline: runs smallest and reports", s == 200 and hd["X-Cale-Goal-Met"] == "false", dict(hd))
    s, hd, r = chat("gemma-3n:auto", key, max_tokens=24, cale_deadline_s=0.2, cale_on_miss="refuse")
    check("impossible deadline + refuse: 422 with estimate", s == 422 and "predicted_s" in r["error"]["cale"], r)

    # 7. a slower computer (debug switch) picks a smaller size for the same deadline.
    # The deadline comes from this computer: 1.5 times what the full size is predicted to need.
    s, hd, r = chat("gemma-3n:auto", other, max_tokens=60, cale_deadline_s=3600, cale_allow_disk=True)
    deadline = round(float(hd["X-Cale-Predicted-S"]) * 1.5, 1)
    s, hd, r = chat("gemma-3n:auto", other, max_tokens=60, cale_deadline_s=deadline, cale_allow_disk=True)
    fast_pct = int(hd["X-Cale-Size"])
    call("/cale/debug/slow", {"factor": 3.0})
    s, hd, r = chat("gemma-3n:auto", other, max_tokens=60, cale_deadline_s=deadline, cale_allow_disk=True)
    slow_pct = int(hd["X-Cale-Size"])
    call("/cale/debug/slow", {"factor": 1.0})
    check(f"slower computer picks a smaller size (deadline {deadline} s)", slow_pct < fast_pct, (fast_pct, slow_pct))

    # 8. models endpoint shows this computer
    m = call("/v1/models", key=key)[2]["cale"]
    check("/v1/models shows this computer", m["this_mac"] is not None and "e2b" in m["this_mac"]["sizes"])

# 9. leaving the block ends the hold; with no other hold the implicit hold remains (the chat calls above), then idles out
s = wait_for(lambda st: not st["holds"])
check("hold ended after the block", not s["holds"])

# 10. memory goal (first wait for the implicit hold of the chat calls above to idle out)
wait_for(lambda st: not st["implicit_hold"], 30)
sizes = call("/v1/models", key=key)[2]["cale"]["this_mac"]["sizes"]
total_gb = oslayer.total_ram() / 1e9
WORK_GB = WORK_BYTES / 1e9
want = next(v for v in sizes.values() if v["size_pct"] == 75)       # a goal that fits the 75% size and not the next one up
pct75 = round((want["ram_gb"] + WORK_GB + 0.02) / total_gb * 100, 2)
with hold(a.base + "/v1", key, memory_pct=60) as h:
    check("memory 60%: full size", h.info["size_pct"] == 100 and h.info["goal_met"], h.info)
with hold(a.base + "/v1", key, memory_pct=pct75) as h:
    check(f"memory {pct75}%: the 75% size", h.info["size_pct"] == 75 and h.info["goal_met"], h.info)
    check("memory goal: really under budget", h.info["ram_gb"] <= h.info["budget_gb"], h.info)
    check("predicted RAM of the 75% size is right (never under, at most 0.3 GB over)", predicted_ok(h.info["ram_gb"], want["ram_gb"]), (h.info["ram_gb"], want["ram_gb"]))
with hold(a.base + "/v1", key, memory_pct=1) as h:
    check("memory 1%: smallest, goal not met", h.info["size_pct"] == MIN_PCT and not h.info["goal_met"], h.info)
    check("predicted RAM of the smallest size is right (never under, at most 0.3 GB over)", predicted_ok(h.info["ram_gb"], sizes["e2b"]["ram_gb"]), (h.info["ram_gb"], sizes["e2b"]["ram_gb"]))
try:
    with hold(a.base + "/v1", key, memory_pct=1, on_miss="refuse"):
        check("memory 1% + refuse", False)
except CaleError as e:
    check("memory 1% + refuse: 422 with GB needed", e.status == 422 and "needs_gb" in e.body, e.body)

# 10b. critical memory pressure does not free a held size
with hold(a.base + "/v1", key, size="fast") as h:
    time.sleep(32)                                                  # the engine ignores pressure for 30 s after a load
    before = status()["resident_gb"]
    call("/cale/debug/pressure", {"level": 4})
    time.sleep(5)
    s = status()
    call("/cale/debug/pressure", {"level": None})
    check("critical pressure: hold and memory stay", len(s["holds"]) == 1 and abs(s["resident_gb"] - before) < RAM_TOL, (before, s["resident_gb"], len(s["holds"])))

# 10c. warn pressure caps the size a request runs at. It must not break a hold (the model must stay loaded, hold intact)
with hold(a.base + "/v1", key, size="75") as h:
    time.sleep(32)                                                  # the engine ignores pressure for 30 s after a load
    call("/cale/debug/pressure", {"level": 2})
    s = wait_for(lambda st: st["cap_pct"] < 100, 10)
    time.sleep(1)
    s = status()
    call("/cale/debug/pressure", {"level": None})
    failed = [e for e in s["events"] if "failed" in e]
    check("warn pressure with a hold: model stays loaded, hold intact, cap applied",
          s["loaded"] and len(s["holds"]) == 1 and s["cap_pct"] == 60 and not failed, (s["loaded"], s["cap_pct"], s["holds"], failed))
    st, hd, r = chat("gemma-3n:smart", key, max_tokens=16)
    check("under the cap a 100% request runs at the cap", st == 200 and int(hd["X-Cale-Size"]) <= 62, hd.get("X-Cale-Size"))

# 11. an app that dies loses its hold within 5 seconds
code = f"""
import sys; sys.path.insert(0, {str(ROOT)!r})
from cale.client import hold
import time
with hold({a.base + '/v1'!r}, {other!r}, size="fast"):
    print("held", flush=True); time.sleep(600)
"""
p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
p.stdout.readline()
check("child holds", len(status()["holds"]) == 1)
p.kill()
t = time.time()
s = wait_for(lambda st: not st["holds"], 8)
check("killed app: hold freed within 5 s", not s["holds"] and time.time() - t < 5, round(time.time() - t, 1))

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
