"""The activity view for the Cale Hub, against a running Cale (start one with scripts/cale_dev_server.sh HOME).

usage: python cale/tests/activity.py [--base http://127.0.0.1:47899] --home HOME
The Hub key is made from the same home folder the server uses, so pass the same HOME.
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
ap.add_argument("--home", required=True)
a = ap.parse_args()
os.environ["CALE_HOME"] = a.home
from cale import keys  # noqa: E402

FAIL = []
SECRET = "zebra-umbrella-7731"                 # must never come back from /cale/activity


def call(path, body=None, key=None):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(a.base + path, data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        r = urllib.request.urlopen(req, timeout=900)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


hub = keys.hub_key()
st, body = call("/cale/connect", {"app": "Activity Test", "wait_s": 60})
check("a normal app gets a key", st == 200 and "api_key" in body, f"{st} {body}")
app_key = body.get("api_key")
st, body = call("/cale/connect", {"app": "cale hub"})
check("the name 'Cale Hub' is reserved", st == 400 and body["error"]["code"] == "reserved_name", f"{st} {body}")
check("no key: 401", call("/cale/activity")[0] == 401)
st, body = call("/cale/activity", key=app_key)
check("an ordinary app is refused: 403 hub_only", st == 403 and body["error"]["code"] == "hub_only", f"{st} {body}")
check("the Hub key is stable and kept", keys.hub_key() == hub)

# one long streamed answer, watched while it runs
seen = {}


def ask():
    req = urllib.request.Request(a.base + "/v1/chat/completions", headers={"Content-Type": "application/json", "Authorization": f"Bearer {app_key}"},
                                 data=json.dumps({"model": "gemma-3n:fast", "stream": True, "max_tokens": 120, "messages": [
                                     {"role": "user", "content": f"Secret word: {SECRET}. Count from 1 to 60, one number per line."}]}).encode())
    with urllib.request.urlopen(req, timeout=900) as r:
        seen["raw"] = r.read()


t = threading.Thread(target=ask)
t.start()
live = None
for _ in range(120):
    time.sleep(0.5)
    st, snap = call("/cale/activity", key=hub)
    run = [r for r in snap.get("running", []) if r["app"] == "Activity Test"]
    if run and run[0]["pieces"] >= 3:
        live = run[0]
        break
t.join()
check("a running call shows its progress", live is not None and live["pieces"] >= 3, str(live))
check("a running call knows its size", live is not None and live["size_pct"] is not None, str(live))
st, snap = call("/cale/activity", key=hub)
mine = [r for r in snap["recent"] if r["app"] == "Activity Test"]
check("the finished call is listed", st == 200 and len(mine) == 1, str(snap))
if mine:
    r = mine[0]
    check("it has tokens, seconds, tokens/s, first token", all(r.get(k) for k in ("tokens", "seconds", "tokens_per_s", "first_token_s")), str(r))
    check("outcome is ok", r["outcome"] == "ok", str(r))
check("nothing is running now", not [r for r in snap["running"] if r["app"] == "Activity Test"], str(snap["running"]))
check("no prompt or answer text in the view", SECRET not in json.dumps(snap) and "zebra" not in json.dumps(snap).lower())

# a failed call is listed as an error
st, _ = call("/v1/chat/completions", {"model": "gemma-3n:fast", "messages": [{"role": "user", "content": "hi"}], "no_such_field": 1}, app_key)
st2, snap = call("/cale/activity", key=hub)
check("a refused request does not stay 'running'", not [r for r in snap["running"] if r["app"] == "Activity Test"], str(snap["running"]))

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
