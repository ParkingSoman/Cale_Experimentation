"""Several apps at once, against a running Cale (start one with scripts/cale_dev_server.py HOME).
When the engine can run answers together (`slots` in /cale/status above 1), they must share the work: nobody waits for another's whole
answer, and the total speed goes up. When it cannot (a Mac with MLX), answers run in turn and must still all be right.

usage: python cale/tests/crowd.py [--base http://127.0.0.1:47899]
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nest.levels3n import load_levels, pct_range  # noqa: E402

MIN_PCT, MAX_PCT = pct_range(load_levels())
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
a = ap.parse_args()
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def call(path, body=None, key=None, timeout=900):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(a.base + path, data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, dict(r.headers), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read())


def chat(key, prompt, **kw):
    return call("/v1/chat/completions", {"model": kw.pop("model", "gemma-3n:smart"), "messages": [{"role": "user", "content": prompt}], "temperature": 0, **kw}, key)


def together(jobs):
    """Run every job (a function) in its own thread at the same time. Returns their results and how long each took."""
    out, took = [None] * len(jobs), [0.0] * len(jobs)

    def run(i):
        t = time.time()
        out[i] = jobs[i]()
        took[i] = time.time() - t
    ths = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
    t0 = time.time()
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    return out, took, time.time() - t0


keys = [call("/cale/connect", {"app": f"Concurrent App {i}"})[2]["api_key"] for i in range(4)]
slots = call("/cale/status")[2].get("slots") or 1
print(f"slots: {slots}", flush=True)
PROMPTS = ["Name three planets and one fact about each.", "Write two sentences about the sea.", "List four fruits with their colors.",
           "Explain in two sentences why the sky is blue."]

# 1. what each prompt answers alone
alone, alone_s = [], []
for i, p in enumerate(PROMPTS):
    t = time.time()
    s, hd, r = chat(keys[i], p, max_tokens=40)
    alone.append(r["choices"][0]["message"]["content"])
    alone_s.append(time.time() - t)

# 2. two and four at once: every answer is complete and (greedy) the same as alone
for n in (2, 4):
    res, took, total = together([lambda i=i: chat(keys[i], PROMPTS[i], max_tokens=40) for i in range(n)])
    ok = all(r[0] == 200 and r[2]["choices"][0]["message"]["content"].strip()[:30] == alone[i].strip()[:30] for i, r in enumerate(res))
    check(f"{n} apps at once: all answered, each as it answers alone", ok, [(r[0], r[2]["choices"][0]["message"]["content"][:30]) for r in res])
    if slots > 1:
        shared = max(r[2]["cale"]["answers_sharing_the_step"] for r in res)
        check(f"{n} apps at once: they shared the steps", shared > 1.2, shared)
        if n == 4:
            check("4 apps at once finish sooner than one after another (the weights are read once per step)", total < 0.85 * sum(alone_s), (round(total, 1), round(sum(alone_s), 1)))

# 3. nobody waits for another's whole answer
if slots > 1:
    long_res, short_res = {}, {}
    tl = threading.Thread(target=lambda: long_res.update(r=chat(keys[0], "Write a long story about a dragon.", max_tokens=200)))
    t0 = time.time()
    tl.start()
    time.sleep(1.0)
    ts = time.time()
    s, hd, r = chat(keys[1], "Say hi.", max_tokens=8)
    short_took = time.time() - ts
    tl.join()
    long_took = time.time() - t0
    check("a short answer does not wait for a long one", s == 200 and short_took < 0.6 * long_took, (round(short_took, 1), round(long_took, 1)))
    check("...and its first token was not delayed by it", float(hd["X-Cale-Waited-S"]) < 1.0, hd.get("X-Cale-Waited-S"))

# 4. goals under a crowd: each app's goal is judged as if the step is shared
def goal_call(i):
    return chat(keys[i], PROMPTS[i], max_tokens=40, model="gemma-3n:auto", cale_min_tokens_per_s=0.1)
res, took, total = together([lambda i=i: goal_call(i) for i in range(3)])
check("3 apps with goals at once: all answered and each reply says what happened", all(r[0] == 200 and r[2]["cale"]["goals_actual"]["speed"]["met"] is True for r in res), [r[0] for r in res])

# 5. an app that leaves does not hurt the others
done = {}
def stayer():
    done["r"] = chat(keys[0], "Write a long story about a fox.", max_tokens=120)
def leaver():
    import http.client
    c = http.client.HTTPConnection(a.base.split("//")[1])
    c.request("POST", "/v1/chat/completions", json.dumps({"model": "gemma-3n:smart", "stream": True, "max_tokens": 200,
              "messages": [{"role": "user", "content": "Write a long story about a bear."}]}), {"Content-Type": "application/json", "Authorization": f"Bearer {keys[1]}"})
    resp = c.getresponse()
    resp.read(200)
    c.close()                                   # leaves in the middle of the answer
th1, th2 = threading.Thread(target=stayer), threading.Thread(target=leaver)
th1.start(); th2.start(); th1.join(); th2.join()
r = done["r"]
check("one app leaves in the middle: the other still gets its whole answer", r[0] == 200 and r[2]["usage"]["completion_tokens"] > 60, r[2].get("usage"))
time.sleep(1.0)
s, hd, r = chat(keys[2], "Say hi.", max_tokens=8)
check("...and the engine is healthy afterwards", s == 200 and r["choices"][0]["message"]["content"], r)
st = call("/cale/status")[2]
check("...and nothing is left running", st.get("last_error") is None or "Cancel" not in str(st.get("last_error")), st.get("last_error"))

# 6. exact sizes are kept: two apps that name different sizes both get theirs
res, took, total = together([lambda: chat(keys[0], PROMPTS[0], max_tokens=30, model="gemma-3n:smart"),
                              lambda: chat(keys[1], PROMPTS[1], max_tokens=30, model="gemma-3n:fast")])
check("an exact size is kept even next to another exact size", res[0][1]["X-Cale-Size"] == str(MAX_PCT) and res[1][1]["X-Cale-Size"] == str(MIN_PCT), [r[1].get("X-Cale-Size") for r in res])

# 7. guided output and tools in a crowd
schema = {"type": "object", "properties": {"city": {"type": "string"}, "n": {"type": "integer"}}, "required": ["city", "n"]}
fmt = {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}
res, took, total = together([lambda i=i: chat(keys[i], f"Pick a city and a number {i}.", max_tokens=60, response_format=fmt) for i in range(3)] +
                            [lambda: chat(keys[3], "Write one sentence about the moon.", max_tokens=30)])
ok = True
for r in res[:3]:
    try:
        d = json.loads(r[2]["choices"][0]["message"]["content"])
        ok = ok and isinstance(d["city"], str) and isinstance(d["n"], int)
    except Exception:
        ok = False
check("three guided answers and one free answer at once: the guided ones are valid", ok and res[3][0] == 200, [r[2]["choices"][0]["message"]["content"][:60] for r in res])

# 8. an answer that cannot be completed fails alone: the app next to it is not hurt
impossible = {"type": "object", "properties": {"a": {"type": "string", "minLength": 1, "maxLength": 0}}, "required": ["a"]}
bad_fmt = {"type": "json_schema", "json_schema": {"name": "bad", "schema": impossible}}
res, took, total = together([lambda: chat(keys[0], "Write two sentences about rivers.", max_tokens=60),
                             lambda: chat(keys[1], "Give me a value.", max_tokens=30, response_format=bad_fmt)])
check("an impossible schema: its app gets a clear 422", res[1][0] == 422 and res[1][2]["error"]["code"] == "constraint_dead_end", res[1])
check("...and the app answering next to it is not hurt", res[0][0] == 200 and res[0][2]["usage"]["completion_tokens"] > 10, res[0])
s, hd, r = chat(keys[2], "Say hi.", max_tokens=8)
check("...and the engine still answers", s == 200 and r["choices"][0]["message"]["content"], r)

# 9. answers that together need more than the shared context are not a crash: they wait, or end with finish_reason length
call("/cale/debug/pool", {"tokens": 1500})                         # a small shared pool (only with a debug server and an engine that has one)
page = "Fact: the sea is wide and deep. " * 60                      # about 600 tokens
res, took, total = together([lambda i=i: chat(keys[i], page + f" Question {i}: what is wide?", max_tokens=24) for i in range(4)])
call("/cale/debug/pool", {"tokens": None})
check("four 600-token prompts in a pool of 1500: all answered, none crashed", all(r[0] == 200 and r[2]["choices"][0]["message"]["content"] for r in res), [(r[0], r[2].get("error")) for r in res])
s, hd, r = chat(keys[0], "Say hi.", max_tokens=8)
check("...and the engine still answers", s == 200, (s, r))

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
