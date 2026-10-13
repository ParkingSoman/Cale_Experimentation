"""Cale says what it did, and refuses what it cannot do. Against a running Cale (scripts/cale_dev_server.py HOME).

usage: python cale/tests/honest.py [--base http://127.0.0.1:47899]
Nothing here belongs to one computer: limits and sizes come from the machine under test.
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
from cale.client import model  # noqa: E402
from nest.levels3n import load_levels, pct_range  # noqa: E402

MIN_PCT, MAX_PCT = pct_range(load_levels())
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
a = ap.parse_args()
FAIL = []


def call(path, body=None, key=None):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(a.base + path, data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        r = urllib.request.urlopen(req, timeout=900)
        return r.status, dict(r.headers), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read())


def chat(key, prompt="Say hello in three words.", model_name="gemma-3n:fast", **kw):
    return call("/v1/chat/completions", {"model": model_name, "messages": [{"role": "user", "content": prompt}], **kw}, key)


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:300]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def err(r):
    return r[0], r[2].get("error", {}).get("code"), r[2].get("error", {}).get("param")


key = call("/cale/connect", {"app": "Honest Test"})[2]["api_key"]
key2 = call("/cale/connect", {"app": "Honest Test Two"})[2]["api_key"]

# 1. A field Cale cannot use is refused by name. It is never ignored in silence.
check("legacy functions -> 400 unsupported_parameter", err(chat(key, functions=[{"name": "f"}])) == (400, "unsupported_parameter", "functions"))
check("logprobs -> 400 unsupported_parameter", err(chat(key, logprobs=True)) == (400, "unsupported_parameter", "logprobs"))
check("tool_choice without tools -> 400", err(chat(key, tool_choice="auto")) == (400, "invalid_tool_choice", "tool_choice"))
check("a tool without a name -> 400", err(chat(key, tools=[{"type": "function", "function": {}}]))[:2] == (400, "invalid_tools"))
check("response_format json_schema without a schema -> 400", err(chat(key, response_format={"type": "json_schema", "json_schema": {}})) == (400, "invalid_response_format", "response_format"))
check("response_format text is fine", chat(key, max_tokens=8, response_format={"type": "text"})[0] == 200)
check("unknown field -> 400 unknown_parameter", err(chat(key, banana=1)) == (400, "unknown_parameter", "banana"))
check("a cale_ field Cale does not have -> 400", err(chat(key, cale_banana=1)) == (400, "unknown_parameter", "cale_banana"))
check("n=2 -> 400", err(chat(key, n=2))[:2] == (400, "unsupported_parameter"))
check("presence_penalty 0.5 -> 400", err(chat(key, presence_penalty=0.5))[:2] == (400, "unsupported_parameter"))
check("fields set to null are ignored", chat(key, max_tokens=8, tools=None, response_format=None)[0] == 200)
check("harmless fields (user, stream_options) are accepted", chat(key, max_tokens=8, user="u1", stream_options={"include_usage": True})[0] == 200)
check("top_p 2 -> 400", err(chat(key, top_p=2))[0] == 400)
check("seed 1.5 -> 400", err(chat(key, seed=1.5)) == (400, "invalid_seed", "seed"))
check("stop with 5 strings -> 400", err(chat(key, stop=["a", "b", "c", "d", "e"])) == (400, "invalid_stop", "stop"))
check("max_tokens error names max_tokens", "max_tokens" in chat(key, max_tokens=99999)[2]["error"]["message"], chat(key, max_tokens=99999)[2])

# 2. stop and seed do what they say
count = "Count from 1 to 12, separated by commas, nothing else."
full = chat(key, count, max_tokens=60)[2]["choices"][0]["message"]["content"]
s, h, j = chat(key, count, max_tokens=60, stop=["6"])
check("stop: the answer ends before the stop text", s == 200 and "6" not in j["choices"][0]["message"]["content"] and "5" in j["choices"][0]["message"]["content"], j)
check("stop: finish_reason is stop", j["choices"][0]["finish_reason"] == "stop", j)
check("stop given as a plain string works", "6" not in chat(key, count, max_tokens=60, stop="6")[2]["choices"][0]["message"]["content"])
t = lambda **k: chat(key, "Write one short sentence about the sea.", max_tokens=30, temperature=1.5, **k)[2]["choices"][0]["message"]["content"]
check("seed: the same seed repeats", t(seed=11) == t(seed=11))
check("seed: another seed differs", t(seed=11) != t(seed=12))
check("no seed: answers differ", t() != t())

# 3. A prompt that does not fit gets a clear 400. The limit is public.
models = call("/v1/models", key=key)[2]
limit = models["cale"]["max_context_tokens"]
check("/v1/models shows the context limit", isinstance(limit, int) and limit >= 4096, models["cale"].get("max_context_tokens"))
long_text = "Item number one two three four five six seven eight nine ten. " * (limit // 8)       # clearly longer than the limit
r = chat(key, long_text, max_tokens=10)
check("prompt longer than the limit -> 400 context_length_exceeded", err(r)[:2] == (400, "context_length_exceeded"), r[2])
check("...and the reply says the numbers", r[2].get("error", {}).get("cale", {}).get("max_context_tokens") == limit, r[2])
mid = "Item number one two three four five six seven eight nine ten. " * 160                  # about 2k tokens: fits by itself, quick even on a slow computer
r = chat(key, mid, max_tokens=limit - 100)
check("prompt + a huge max_tokens that cannot fit -> 400", err(r)[:2] == (400, "context_length_exceeded"), r[2])
check("a prompt that fits is answered", chat(key, mid, max_tokens=5)[0] == 200)
check("engine still healthy after those refusals", call("/cale/status")[2].get("last_error") is None)

# 4. An engine failure is reported, the engine resets, the next call works
if call("/cale/debug/fail", {})[0] == 200:
    r = chat(key, max_tokens=5)
    check("engine failure -> 500 engine_error", err(r)[:2] == (500, "engine_error"), r)
    st = call("/cale/status")[2]
    check("status says that something failed, without the details", st.get("last_error") and "RuntimeError" in st["last_error"] and "injected" not in st["last_error"], st)
    check("the error reply gives no details either", "injected" not in json.dumps(r[2]), r)
    r = chat(key, max_tokens=5)
    check("the next call works (model reloads)", r[0] == 200 and r[2]["choices"][0]["message"]["content"], r)
else:
    print("SKIP engine failure test (server not in debug mode)")

# 5. Time spent waiting in line counts. The reply says so, and does not claim a goal was met when it was not.
# Answers run together when the engine has slots, so fill every slot first: then the next call must wait.
n_slots = call("/cale/status")[2].get("slots") or 1
done = {}
def long_call(i):
    done[i] = chat(key2, f"Write a long story about a dragon, version {i}.", max_tokens=250)
threads = [threading.Thread(target=long_call, args=(i,)) for i in range(n_slots)]
for th in threads:
    th.start()
time.sleep(1.5)
r = chat(key, "Say hi.", model_name="gemma-3n:auto", max_tokens=8, cale_max_first_token_s=0.5)
for th in threads:
    th.join()
cale = r[2]["cale"]
check("a call that waited reports waited_s", cale.get("waited_s", 0) >= 1.0, cale)
check("first-token goal missed while waiting in line: goal_met_actual is false", cale.get("goal_met_actual") is False, cale)
check("...and the header agrees", r[1].get("X-Cale-Goal-Met") == "false", r[1])
check("...and total_s includes the wait", cale.get("total_s", 0) >= cale.get("waited_s", 99), cale)

# 6. A memory goal caps what the object's calls run at, even right after another hold ended
with model(a.base + "/v1", key, name="memcap", memory_pct=1) as m:
    r = chat(key, model_name=m.id, max_tokens=8)
    check("memory goal missed: the call runs at the size the object reports", r[1].get("X-Cale-Size") == str(m.info["size_pct"]), (r[1], m.info))
    check("...and says the memory goal was not met", r[1].get("X-Cale-Memory-Goal-Met") == "false", r[1])

# 7. A client that leaves stops its work: the next app is not stuck behind an abandoned request
import socket
from urllib.parse import urlparse
u = urlparse(a.base)


def abandon(prompt, max_tokens, after_s):
    body = json.dumps({"model": "gemma-3n:smart", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}).encode()
    s = socket.create_connection((u.hostname, u.port))
    s.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nAuthorization: Bearer {key}\r\n"
               f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body)
    time.sleep(after_s)
    s.close()


def quick():
    t0 = time.time()
    r = chat(key2, "Say hi.", model_name="gemma-3n:fast", max_tokens=8)
    return r[0], time.time() - t0


q0 = quick()[1]
abandon("Write a very long story about a dragon.", 1000, 1.5)        # about 30 s of work on a fast computer, minutes on a slow one
st, q1 = quick()
check("answer phase: the next app is not stuck behind an abandoned answer", st == 200 and q1 < q0 + 8, f"alone {q0:.1f}s, after abandon {q1:.1f}s")
filler = "Item number one two three four five six seven eight nine ten. " * 220      # about 3k tokens: seconds to minutes of prompt reading
abandon(filler, 50, 1.0)
st, q2 = quick()
check("prompt phase: the next app is not stuck behind an abandoned prompt", st == 200 and q2 < q0 + 8, f"alone {q0:.1f}s, after abandon {q2:.1f}s")

print("\nall passed" if not FAIL else f"\nFAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
