"""Any language, any computer setup: approval without a screen, plain-HTTP leases, the `cale` command, curl, and the Node client.
Needs a test server with no screen: scripts/run_suites.py starts one (CALE_HEADLESS=1, CALE_DEV_APPROVE=0, CALE_TEST_HOME set).

usage: python cale/tests/anylang.py [--base http://127.0.0.1:47899]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
a = ap.parse_args()
HOME = os.environ.get("CALE_TEST_HOME")
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def call(path, body=None, key=None, method=None):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(a.base + path, data=None if body is None else json.dumps(body).encode(), headers=h, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=300)
        return r.status, dict(r.headers), json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read() or b"{}")


def cale_cli(*args, tty=False, env_extra=None):
    """The `cale` command, as a person would run it on this computer."""
    env = dict(os.environ, CALE_HOME=HOME, CALE_PORT=a.base.rsplit(":", 1)[1], PYTHONPATH=str(ROOT), **(env_extra or {}))
    return subprocess.run([sys.executable, "-m", "cale.cli", *args], capture_output=True, text=True, env=env, cwd=ROOT, timeout=120, stdin=subprocess.DEVNULL)


if not HOME:
    print("SKIP: CALE_TEST_HOME is not set (run this suite with scripts/run_suites.py)")
    sys.exit(0)

# 1. approval without a screen
s, hd, r = call("/cale/connect", {"app": "Headless App", "wait_s": 0})
check("connect with no screen and wait_s 0: 202 pending, says how to allow", s == 202 and r["status"] == "pending" and "cale allow" in r["message"], (s, r))
ticket = r.get("ticket")
s, hd, r = call(r["poll"])
check("polling: still pending", s == 202 and r["status"] == "pending", (s, r))
out = cale_cli("pending")
check("`cale pending` lists the app", "Headless App" in out.stdout, out.stdout + out.stderr)
out = cale_cli("allow", "Headless App")
check("`cale allow` refuses when no person is at a terminal", out.returncode == 2 and "terminal" in out.stdout, out.stdout + out.stderr)
out = cale_cli("allow", "Nobody", env_extra={"CALE_ALLOW_NON_INTERACTIVE": "1"})
check("`cale allow` for an app that is not waiting says so", out.returncode == 1 and "No app named" in out.stdout, out.stdout)
out = cale_cli("allow", "Headless App", env_extra={"CALE_ALLOW_NON_INTERACTIVE": "1"})
check("`cale allow` (test mode) allows", out.returncode == 0 and "Allowed" in out.stdout, out.stdout + out.stderr)
for _ in range(20):
    s, hd, r = call("/cale/connect/status?ticket=" + ticket)
    if s != 202:
        break
    time.sleep(0.5)
key = r.get("api_key")
check("after allow: the ticket gives the key", s == 200 and key and r["app"] == "Headless App", (s, r))
s, hd, r = call("/cale/connect/status?ticket=" + ticket)
check("...once only", s == 404, (s, r))
s, hd, r = call("/v1/models", key=key)
check("the key works", s == 200 and r["cale"]["this_computer"] is not None or s == 200, (s, str(r)[:100]))

s, hd, r = call("/cale/connect", {"app": "Denied App", "wait_s": 0})
t2 = r["ticket"]
out = cale_cli("deny", "Denied App", env_extra={"CALE_ALLOW_NON_INTERACTIVE": "1"})
for _ in range(20):
    s, hd, r = call("/cale/connect/status?ticket=" + t2)
    if s != 202:
        break
    time.sleep(0.5)
check("`cale deny`: the app gets 403 denied", s == 403 and r["error"]["code"] == "denied", (s, r))

t0 = time.time()
s, hd, r = call("/cale/connect", {"app": "Nobody Answers", "wait_s": 30})
check("nobody answers: the request ends as 'no' by itself (here after 8 s)", s == 403 and time.time() - t0 < 25, (s, round(time.time() - t0, 1)))

s, hd, r = call("/cale/connect", {"app": "Blocking App", "wait_s": 3})
check("a blocking connect that is not answered in time gets 202 with a ticket", s == 202 and r.get("ticket"), (s, r))

# 2. leases: plain HTTP, no open connection
s, hd, r = call("/cale/models", {"name": "Leased", "size": "fast", "max_tokens": 12, "lease_s": 6}, key)
check("a model object on a lease: the reply comes at once, with the lease", s == 200 and r["id"] == "leased" and r["lease"]["expires_in_s"] == 6, (s, r))
oid = r["id"]
ids = [m["id"] for m in call("/v1/models", key=key)[2]["data"]]
check("...and it is listed", oid in ids, ids)
time.sleep(4)
s, hd, r = call("/v1/chat/completions", {"model": oid, "messages": [{"role": "user", "content": "Say hi."}]}, key)
check("using the object renews its lease", s == 200 and r["choices"][0]["message"]["content"], (s, r))
time.sleep(4)                                                     # 8 s since creation, 4 s since the last use: still alive
s, hd, r = call("/v1/chat/completions", {"model": oid, "messages": [{"role": "user", "content": "Say hi."}]}, key)
check("...so it is still alive after the first lease would have ended", s == 200, (s, r))
time.sleep(9)
s, hd, r = call("/v1/chat/completions", {"model": oid, "messages": [{"role": "user", "content": "Say hi."}]}, key)
check("an object nobody uses ends by itself", s == 404, (s, r))

s, hd, r = call("/cale/models", {"name": "Ends", "size": "fast", "lease_s": 60}, key)
s2, hd, r2 = call("/cale/lease", {"id": "ends"}, key)
check("renew by hand", s2 == 200 and r2["expires_in_s"] == 60, (s2, r2))
s3, hd, r3 = call("/cale/models/ends", key=key, method="DELETE")
check("DELETE ends it", s3 == 200 and r3.get("ended"), (s3, r3))
s4, hd, r4 = call("/cale/lease", {"id": "ends"}, key)
check("...and it is gone", s4 == 404, (s4, r4))

key_b = call("/cale/connect", {"app": "Other Headless"})[0]
s, hd, r = call("/cale/models", {"name": "Mine", "size": "fast", "lease_s": 30}, key)
oid = r["id"]
s2, hd, r2 = call("/cale/lease", {"id": oid, "end": True}, "cale-not-a-key")
check("a lease cannot be ended with a wrong key", s2 == 401, (s2, r2))
call("/cale/lease", {"id": oid, "end": True}, key)

s, hd, r = call("/cale/load", {"size": "fast", "lease_s": 6}, key)
hid = r.get("hold")
check("a hold on a lease: reply at once", s == 200 and hid and r["lease"], (s, r))
st = call("/cale/status")[2]
check("...and it is listed", any(h["hold"] == hid for h in st["holds"]), st["holds"])
time.sleep(9)
st = call("/cale/status")[2]
check("a hold nobody renews ends by itself", not any(h["hold"] == hid for h in st["holds"]), st["holds"])
s, hd, r = call("/cale/load", {"size": "fast", "lease_s": 60}, key)
s2, hd, r2 = call("/cale/lease", {"hold": r["hold"], "end": True}, key)
check("a hold's lease can be ended", s2 == 200 and r2.get("ended"), (s2, r2))
s, hd, r = call("/cale/models", {"name": "Bad", "size": "fast", "lease_s": 1}, key)
check("a lease must be 5 to 3600 s", s == 400, (s, r))

# 3. the `cale` command
out = cale_cli("status", "--json")
d = json.loads(out.stdout) if out.stdout.strip().startswith("{") else {}
check("`cale status --json` for programs", out.returncode == 0 and d.get("running") is True, out.stdout + out.stderr)
out = cale_cli("computer")
check("`cale computer` shows the sizes and the floor", out.returncode == 0 and "answer tok/s" in out.stdout and "The most this computer can do" in out.stdout, out.stdout[-300:] + out.stderr[-200:])
out = cale_cli("apps")
check("`cale apps` lists the app we allowed", "Headless App" in out.stdout, out.stdout)

# 4. curl: no library, no Python on the app's side
if shutil.which("curl"):
    base = a.base
    c = subprocess.run(["curl", "-s", "-X", "POST", base + "/cale/connect", "-H", "Content-Type: application/json", "-d", '{"app":"Curl App","wait_s":0}'], capture_output=True, text=True)
    cd = json.loads(c.stdout)
    check("curl: connect returns a ticket", cd.get("status") == "pending", c.stdout)
    cale_cli("allow", "Curl App", env_extra={"CALE_ALLOW_NON_INTERACTIVE": "1"})
    time.sleep(1.5)
    c = subprocess.run(["curl", "-s", base + cd["poll"]], capture_output=True, text=True)
    ck = json.loads(c.stdout).get("api_key")
    check("curl: the ticket gives a key", bool(ck), c.stdout)
    c = subprocess.run(["curl", "-s", "-X", "POST", base + "/v1/chat/completions", "-H", f"Authorization: Bearer {ck}", "-H", "Content-Type: application/json",
                        "-d", json.dumps({"model": "gemma-3n:auto", "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 12, "cale_min_tokens_per_s": 0.1})],
                       capture_output=True, text=True)
    cj = json.loads(c.stdout)
    check("curl: a chat with a goal, and the reply says what happened", cj["choices"][0]["message"]["content"] and cj["cale"]["goals_actual"]["speed"]["met"] is True, c.stdout[:200])
else:
    print("SKIP curl (not installed)")

# 5. the Node client
if shutil.which("node"):
    p = subprocess.run(["node", str(ROOT / "scripts" / "js_client_test.mjs"), a.base, key], capture_output=True, text=True, timeout=600)
    for line in p.stdout.splitlines():
        if line.startswith("PASS ") or line.startswith("FAIL "):
            print(line, flush=True)
            if line.startswith("FAIL "):
                FAIL.append(line[5:].split("  [")[0])
    if p.returncode != 0 and not any(l.startswith("FAIL ") for l in p.stdout.splitlines()):
        check("node client test ran", False, p.stdout[-300:] + p.stderr[-300:])
else:
    print("SKIP node (not installed)")

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
