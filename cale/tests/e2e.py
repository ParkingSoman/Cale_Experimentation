"""End-to-end checks for a running Cale, using the plain `openai` package like a developer would.

  CALE_DEBUG=1 CALE_AUTO_APPROVE=1 on the server lets this run without clicking Allow.

  python cale/tests/e2e.py [--base http://127.0.0.1:47821] [--model]     # --model also runs generation tests
Run it with the repo venv (it has openai):  .venv/bin/python cale/tests/e2e.py --model
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nest.levels3n import load_levels, pct_range, pick  # noqa: E402

LEVELS = load_levels()
MIN_PCT, MAX_PCT = pct_range(LEVELS)


def ran(pct):
    """The model name Cale should report for a request of this percent."""
    p = pick(LEVELS, pct)["pct"]
    return "gemma-3n" if p >= 100 else f"gemma-3n:{p}"


PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)


def http(base, path, body=None, headers=None, method=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:47821")
    ap.add_argument("--model", action="store_true", help="also run tests that load and use the model")
    a = ap.parse_args()
    base = a.base

    s, j = http(base, "/cale/status")
    check("status answers without a key", s == 200 and j.get("ok") is True, j)

    s, j = http(base, "/v1/models")
    check("models without key -> 401", s == 401 and j["error"]["code"] == "invalid_api_key", j)
    s, j = http(base, "/v1/chat/completions", {"model": "gemma-3n", "messages": []}, {"Authorization": "Bearer nope"})
    check("wrong key -> 401", s == 401, j)

    s, j = http(base, "/cale/status", headers={"Host": "evil.example"})
    check("foreign Host header refused", s == 403, j)
    s, j = http(base, "/cale/connect", {"app": "Web Page"}, {"Origin": "https://evil.example"})
    check("connect with an Origin header refused", s == 403 and j["error"]["code"] == "browser_not_allowed", j)
    s, j = http(base, "/cale/connect", {})
    check("connect without app name -> 400", s == 400, j)
    s, j = http(base, "/cale/status", headers={"Origin": "https://evil.example"})
    check("any route refuses an Origin header", s == 403, j)
    s, j = http(base, "/cale/status", headers={"Sec-Fetch-Site": "cross-site"})
    check("any route refuses Sec-Fetch-Site", s == 403, j)

    s, j = http(base, "/cale/connect", {"app": "E2E Test"})
    check("connect returns a key", s == 200 and j.get("api_key", "").startswith("cale-"), j)
    key = j.get("api_key", "")
    auth = {"Authorization": f"Bearer {key}"}

    s, j = http(base, "/v1/models", headers=auth)
    ids = [m["id"] for m in j.get("data", [])]
    check("models lists smart, fast and auto", s == 200 and ids == ["gemma-3n:smart", "gemma-3n:fast", "gemma-3n:auto"], j)
    check("models reports the size range", j["cale"]["sizes"]["min"] == MIN_PCT and j["cale"]["sizes"]["max"] == 100, j)
    s, j = http(base, "/v1/chat/completions", {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}, auth)
    check("unknown model -> 404", s == 404 and j["error"]["code"] == "model_not_found", j)
    s, j = http(base, "/v1/chat/completions", {"model": "gemma-3n:abc", "messages": [{"role": "user", "content": "hi"}]}, auth)
    check("bad size -> 404 model_not_found", s == 404 and j["error"]["code"] == "model_not_found", j)
    s, j = http(base, "/v1/chat/completions", {"model": "gemma-3n", "messages": [{"role": "function", "content": "x"}]}, auth)
    check("unsupported role -> 400", s == 400, j)
    ok_msg = [{"role": "user", "content": "hi"}]
    for field, val in (("max_tokens", -1), ("max_tokens", "100"), ("max_tokens", 10**9), ("temperature", 5), ("temperature", float("nan"))):
        s, j = http(base, "/v1/chat/completions", {"model": "gemma-3n", "messages": ok_msg, field: val}, auth)
        check(f"bad {field}={val!r} -> 400", s == 400, j)
    s, j = http(base, "/v1/chat/completions", {"model": "gemma-3n:nan", "messages": ok_msg}, auth)
    check("size nan -> 404 model_not_found", s == 404 and j["error"]["code"] == "model_not_found", j)
    s, j = http(base, "/v1/chat/completions", {"model": "gemma-3n", "messages": ["hi"]}, auth)
    check("message that is not an object -> 400", s == 400, j)
    s, j = http(base, "/v1/chat/completions", {"model": "gemma-3n", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "x"}}]}]}, auth)
    check("image input -> 400 (not yet)", s == 400, j)

    if a.model:
        from openai import OpenAI, AuthenticationError
        c = OpenAI(base_url=base + "/v1", api_key=key, timeout=600)
        for name, size in (("gemma-3n:smart", 100), ("gemma-3n:75", 75), ("gemma-3n:fast", MIN_PCT)):
            t = time.time()
            r = c.chat.completions.create(model=name, max_tokens=40,
                                          messages=[{"role": "user", "content": "Name the capital of France in one sentence."}])
            txt = r.choices[0].message.content
            check(f"size {size}: answers", "Paris" in txt, txt)
            check(f"size {size}: reports the size that ran", r.model == ran(size), r.model)
            print(f"      {size}%: {time.time() - t:.1f}s  {txt[:70]!r}")
        r = c.chat.completions.create(model="gemma-3n:30", max_tokens=8, messages=[{"role": "user", "content": "Say hi."}])
        check("size below the floor is raised to the smallest", r.model == ran(MIN_PCT), r.model)
        r = c.chat.completions.create(model="gemma-3n:60", max_tokens=60, messages=[
            {"role": "system", "content": "You answer in exactly one word."},
            {"role": "user", "content": "My name is Rivka."}, {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "What is my name?"}])
        check("system + multi-turn history is used", "Rivka" in r.choices[0].message.content, r.choices[0].message.content)
        parts, final_model = [], None
        for ch in c.chat.completions.create(model="gemma-3n:80", max_tokens=40, stream=True,
                                            messages=[{"role": "user", "content": "Count from one to five in words."}]):
            if ch.choices and ch.choices[0].delta.content:
                parts.append(ch.choices[0].delta.content)
            final_model = ch.model
        check("streaming sends several pieces", len(parts) > 3 and "".join(parts).strip() != "", parts[:3])
        check("streaming final chunk reports the size", final_model == ran(80), final_model)
        import concurrent.futures as cf

        def one(i):
            return c.chat.completions.create(model="gemma-3n:60", max_tokens=20,
                                             messages=[{"role": "user", "content": f"Reply with the number {i}."}]).choices[0].message.content
        with cf.ThreadPoolExecutor(2) as ex:
            outs = list(ex.map(one, (7, 9)))
        check("two requests at once both finish", all(o.strip() for o in outs), outs)
        # a second app with its own key uses the same loaded model: memory does not grow
        mem1 = http(base, "/cale/status")[1]["resident_gb"]
        s, j2 = http(base, "/cale/connect", {"app": "E2E App Two"})
        c2 = OpenAI(base_url=base + "/v1", api_key=j2.get("api_key", ""), timeout=600)
        r2 = c2.chat.completions.create(model="gemma-3n:fast", max_tokens=10, messages=[{"role": "user", "content": "Say ok."}])
        st = http(base, "/cale/status")[1]
        check("second app uses the same loaded model", bool(r2.choices[0].message.content.strip()) and st["loaded"]
              and abs(st["resident_gb"] - mem1) < 0.3, (mem1, st))
        for alias, want in (("smart", MAX_PCT), ("fast", MIN_PCT)):
            r = c.chat.completions.create(model=f"gemma-3n:{alias}", max_tokens=4, messages=[{"role": "user", "content": "Hi"}])
            check(f"alias {alias} runs size {want}", r.model == ran(want), r.model)
        try:
            OpenAI(base_url=base + "/v1", api_key="cale-wrong").models.list()
            check("openai client with wrong key is rejected", False)
        except AuthenticationError:
            check("openai client with wrong key is rejected", True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
