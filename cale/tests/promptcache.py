"""The prompt cache against a running Cale: a chat that grows by a turn does not read its whole history again.

usage: python cale/tests/promptcache.py [--base http://127.0.0.1:47899]
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
a = ap.parse_args()
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def call(path, body=None, key=None):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(a.base + path, data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        r = urllib.request.urlopen(req, timeout=900)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def chat(key, msgs, model="gemma-3n:smart", **kw):
    return call("/v1/chat/completions", {"model": model, "messages": msgs, "temperature": 0, "max_tokens": 24, **kw}, key)


key = call("/cale/connect", {"app": "Cache Test"})[1]["api_key"]
DOC = " ".join(f"Fact {i}: the number {i * 7} is written {i * 7} in digits." for i in range(1, 36))
sys_msg = {"role": "system", "content": "You answer questions about this document. " + DOC}
q1 = {"role": "user", "content": "What is Fact 3?"}

s, r1 = chat(key, [sys_msg, q1])
u1 = r1["usage"]
check("first call: nothing cached", s == 200 and u1["prompt_tokens_details"]["cached_tokens"] == 0, u1)
a1 = r1["choices"][0]["message"]["content"]
turn2 = [sys_msg, q1, {"role": "assistant", "content": a1}, {"role": "user", "content": "And what is Fact 4?"}]
s, r2 = chat(key, turn2)
u2 = r2["usage"]
check("second turn: the history is taken from the cache", s == 200 and u2["prompt_tokens_details"]["cached_tokens"] >= 0.7 * u1["prompt_tokens"], (u1, u2))
check("second turn: the first token comes sooner", r2["cale"]["first_token_s"] < 0.8 * r1["cale"]["first_token_s"], (r1["cale"]["first_token_s"], r2["cale"]["first_token_s"]))
a2 = r2["choices"][0]["message"]["content"]

# the answer must be the same as without the cache: break the cache with an unrelated call, then ask the same turn again
chat(key, [{"role": "user", "content": "Say hi."}])
s, r3 = chat(key, turn2)
check("after another conversation: nothing cached", r3["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, r3["usage"])
a3 = r3["choices"][0]["message"]["content"]
check("the answer with the cache equals the answer without it", a2.strip()[:40] == a3.strip()[:40], (a2, a3))

# the cache is kept per size: another size starts again
s, r4 = chat(key, turn2, model="gemma-3n:fast")
check("a different size does not use the cache", r4["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, r4["usage"])
s, r5 = chat(key, turn2 + [{"role": "assistant", "content": "x"}, {"role": "user", "content": "ok"}], model="gemma-3n:fast")
check("...and builds its own", r5["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, r5["usage"])

# one app never benefits from, or learns about, another app's conversation
key_b = call("/cale/connect", {"app": "Cache Test B"})[1]["api_key"]
chat(key, [sys_msg, q1])
s, rb = chat(key_b, [sys_msg, q1, {"role": "assistant", "content": a1}, {"role": "user", "content": "And what is Fact 4?"}])
check("another app with the same start gets nothing from the cache (and learns nothing)", s == 200 and rb["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, rb["usage"])
s, ra = chat(key, [sys_msg, q1])
check("the first app's cache was replaced by the other app's work: it starts again, correctly", s == 200 and ra["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, ra["usage"])

# a changed history is not reused wrongly
changed = [{"role": "system", "content": "You answer questions about this document. " + DOC.replace("Fact 1:", "Item 1:")}, q1]
chat(key, [sys_msg, q1])
s, r6 = chat(key, changed)
check("a changed start: reused only up to the change", s == 200 and r6["usage"]["prompt_tokens_details"]["cached_tokens"] < r6["usage"]["prompt_tokens"] - 5, r6["usage"])

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
