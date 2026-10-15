"""A GPU failure must never reach an app. Against a running Cale started with CALE_DEBUG=1 and CALE_GPU_COOLDOWN_S=8
(scripts/cale_dev_server.sh HOME, or `cale start` with those variables).

usage: python cale/tests/gpu_fallback.py [--base http://127.0.0.1:47899]
The test makes the engine fail inside a step, the way a GPU out-of-memory error does (`/cale/debug/fail` with kind "gpu").
On a computer where Cale runs on the CPU there is no GPU to fall back from: the test says so and passes the part it can.
Expected on a GPU computer: the app gets a normal answer, the engine runs on the CPU for the cooldown, then the GPU is used again.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
ap.add_argument("--cooldown", type=float, default=8.0, help="the CALE_GPU_COOLDOWN_S the server was started with")
a = ap.parse_args()
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:300]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def call(path, body=None, key=None, timeout=900):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(a.base + path, data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def ask(key, text="Say hello in five words."):
    return call("/v1/chat/completions", {"model": "gemma-3n:fast", "max_tokens": 12, "temperature": 0,
                                         "messages": [{"role": "user", "content": text}]}, key)


def placement():
    return str(call("/cale/status")[1].get("placement") or "")


key = call("/cale/connect", {"app": "GPU Fallback Test"})[1]["api_key"]
code, _ = ask(key)
check("warm-up answer", code == 200, code)
start = placement()
print(f"   placement at the start: {start}")
on_gpu = "device=" in start and "device=cpu" not in start
code, body = call("/cale/debug/fail", {"kind": "gpu"})
if code != 200:
    print("   (Cale was not started with CALE_DEBUG=1: cannot inject a failure)")
    check("debug failure available", False, code)
else:
    t = time.time()
    code, body = ask(key)
    text = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") if code == 200 else None
    print(f"   answer after the injected failure: HTTP {code}, {time.time() - t:.1f} s, {text!r}")
    if on_gpu:
        check("the app gets a normal answer after a GPU failure (no error)", code == 200 and bool(text), body)
        now = placement()
        check("the engine now runs on the CPU", "device=cpu" in now, now)
        time.sleep(a.cooldown + 1)
        code, _ = ask(key)
        later = placement()
        check("after the cooldown the answer works and the GPU is used again", code == 200 and "device=cpu" not in later, later)
    else:
        check("on the CPU an injected failure is reported as a normal engine error", code in (200, 500, 502, 503), code)
        code, _ = ask(key)
        check("the next call works", code == 200, code)

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
