"""Log-probabilities of the tokens of an answer (`logprobs`, `top_logprobs`). Against a running Cale (scripts/cale_dev_server.py HOME).

usage: python cale/tests/logprobs.py [--base http://127.0.0.1:47899]
"""
import argparse
import json
import math
import sys
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
a = ap.parse_args()
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:600]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def post(body, key, stream=False):
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        r = urllib.request.urlopen(req, timeout=900)
        if stream:
            return r.status, [json.loads(l[6:]) for l in r.read().decode().splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def chat(key, **kw):
    return post({"model": "gemma-3n:fast", "temperature": 0, "max_tokens": 16, "messages": [{"role": "user", "content": "Name three colours, separated by commas."}], **kw}, key)


key = json.loads(urllib.request.urlopen(urllib.request.Request(a.base + "/cale/connect", data=json.dumps({"app": "Logprobs Test"}).encode(),
                                                               headers={"Content-Type": "application/json"})).read())["api_key"]

s, r = chat(key, logprobs=True, top_logprobs=3)
ok = s == 200
lp = r["choices"][0].get("logprobs", {}).get("content") if ok else None
check("logprobs: a reply with one entry per token", ok and lp and len(lp) == r["usage"]["completion_tokens"], r)
if lp:
    e = lp[0]
    check("every entry has token, logprob, bytes, top_logprobs", all({"token", "logprob", "bytes", "top_logprobs"} <= set(x) for x in lp), e)
    check("log-probabilities are at most 0 and are numbers", all(isinstance(x["logprob"], float) and x["logprob"] <= 1e-6 and math.isfinite(x["logprob"]) for x in lp), [x["logprob"] for x in lp])
    check("top_logprobs: 3 entries, best first", all(len(x["top_logprobs"]) == 3 and x["top_logprobs"][0]["logprob"] >= x["top_logprobs"][1]["logprob"] >= x["top_logprobs"][2]["logprob"] for x in lp), e)
    check("greedy: the chosen token is the best one", all(abs(x["logprob"] - x["top_logprobs"][0]["logprob"]) < 1e-3 for x in lp), [(x["token"], x["logprob"], x["top_logprobs"][0]) for x in lp][:4])
    joined = b"".join(bytes(x["bytes"]) for x in lp).decode("utf-8", "replace")
    check("the tokens add up to the answer", r["choices"][0]["message"]["content"].strip() in joined.strip() or joined.strip().startswith(r["choices"][0]["message"]["content"].strip()[:10]), (joined, r["choices"][0]["message"]["content"]))
    check("probabilities of the top tokens add up to at most 1", all(sum(math.exp(t["logprob"]) for t in x["top_logprobs"]) <= 1.0001 for x in lp))

s, r = chat(key, logprobs=True)
check("top_logprobs left out: entries with an empty top_logprobs", s == 200 and all(x["top_logprobs"] == [] for x in r["choices"][0]["logprobs"]["content"]), r)
s, r = chat(key)
check("logprobs not asked: no logprobs in the reply", s == 200 and "logprobs" not in r["choices"][0], r)

s, chunks = post({"model": "gemma-3n:fast", "temperature": 0, "max_tokens": 16, "stream": True, "logprobs": True, "top_logprobs": 2,
                  "messages": [{"role": "user", "content": "Name three colours, separated by commas."}]}, key, stream=True)
streamed = [x for c in chunks for x in (c["choices"][0].get("logprobs") or {}).get("content", [])] if s == 200 else []
s2, r2 = chat(key, logprobs=True, top_logprobs=2)
whole = r2["choices"][0]["logprobs"]["content"] if s2 == 200 else []
check("streaming: the same entries as the whole reply", s == 200 and [x["token"] for x in streamed] == [x["token"] for x in whole] and len(streamed) > 0, (len(streamed), len(whole)))

err = lambda r: (r[0], r[1].get("error", {}).get("code"))                                           # noqa: E731
check("top_logprobs 21 -> 400", err(chat(key, logprobs=True, top_logprobs=21)) == (400, "invalid_top_logprobs"))
check("top_logprobs without logprobs -> 400", err(chat(key, top_logprobs=2)) == (400, "invalid_top_logprobs"))
check("logprobs 'yes' -> 400", err(chat(key, logprobs="yes")) == (400, "invalid_logprobs"))
tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}
check("logprobs with tools -> 400 unsupported_parameter", err(chat(key, logprobs=True, tools=[tool])) == (400, "unsupported_parameter"))
check("logprobs with response_format -> 400 unsupported_parameter", err(chat(key, logprobs=True, response_format={"type": "json_object"})) == (400, "unsupported_parameter"))

print("\nall passed" if not FAIL else f"\nFAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
