"""Memory-pressure behaviour of a running Cale (needs CALE_DEBUG=1 on the server).

Forces 'warn', checks the cap and the memory drop, checks a 100% request runs at the cap,
then forces 'critical' and checks the model unloads and loads again on the next request.
usage: .venv/bin/python cale/tests/pressure.py --base URL [--recover-wait SECONDS]   (a TEST Cale: never port 47821)
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nest.levels3n import load_levels, pick  # noqa: E402

LEVELS = load_levels()

from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="a TEST Cale on a spare port (scripts/cale_dev_server.py). Never the real one: this test shows an Allow question and forces memory pressure.")
ap.add_argument("--recover-wait", type=int, default=0, help="also wait this long for the cap to return")
a = ap.parse_args()
if a.base.rstrip("/").endswith(":47821"):
    sys.exit("This is the real Cale's port (47821). A test must use a TEST Cale: scripts/cale_dev_server.py HOME, then --base http://127.0.0.1:47899")
FAIL = []


def call(path, body=None):
    req = urllib.request.Request(a.base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())


def status():
    return call("/cale/status")


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def wait_for(pred, secs=40):
    t = time.time()
    while time.time() - t < secs:
        s = status()
        if pred(s):
            return s
        time.sleep(1)
    return status()


key = call("/cale/connect", {"app": "Pressure Test"})["api_key"]
c = OpenAI(base_url=a.base + "/v1", api_key=key, timeout=600)
c.chat.completions.create(model="gemma-3n:60", max_tokens=4, messages=[{"role": "user", "content": "hi"}])
s0 = status()
print(f"start: loaded={s0['loaded']} cap={s0['cap_pct']}% resident={s0['resident_gb']} GB")
time.sleep(max(0, 32 - s0["idle_s"]))                 # the engine ignores pressure for a while after loading

call("/cale/debug/pressure", {"level": 2})
s1 = wait_for(lambda s: s["cap_pct"] <= 60)
print(f"warn:  cap={s1['cap_pct']}% resident={s1['resident_gb']} GB")
check("warn: cap drops to 60%", s1["cap_pct"] == 60, s1)
check("warn: resident memory does not grow", s1["resident_gb"] <= s0["resident_gb"] + 0.3, (s0["resident_gb"], s1["resident_gb"]))
t = time.time()
r = c.chat.completions.create(model="gemma-3n", max_tokens=30, messages=[{"role": "user", "content": "Name the capital of France."}])
under_cap = f"gemma-3n:{pick(LEVELS, 100, 60)['pct']}"
check("warn: a 100% request runs at the largest size under the cap and says so", r.model == under_cap, (r.model, under_cap))
print(f"       answer in {time.time() - t:.1f}s: {r.choices[0].message.content[:60]!r}")
check("warn: the model still answers", len(r.choices[0].message.content.strip()) > 0)

if a.recover_wait:
    call("/cale/debug/pressure", {"level": 1})
    s2 = wait_for(lambda s: s["cap_pct"] == 100, a.recover_wait)
    print(f"normal: cap={s2['cap_pct']}% resident={s2['resident_gb']} GB")
    check("normal: cap grows back", s2["cap_pct"] == 100, s2)

call("/cale/debug/pressure", {"level": 4})
s3 = wait_for(lambda s: not s["loaded"])
print(f"critical: loaded={s3['loaded']}")
check("critical: model unloads", not s3["loaded"], s3)
call("/cale/debug/pressure", {"level": 1})
r = c.chat.completions.create(model="gemma-3n:60", max_tokens=20, messages=[{"role": "user", "content": "Say hello."}])
s4 = status()
check("critical: next request loads the model again and answers", s4["loaded"] and r.choices[0].message.content.strip() != "", s4)
call("/cale/debug/pressure", {"level": None})
print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all passed'}")
