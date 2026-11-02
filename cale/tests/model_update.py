"""POST /cale/models/<id> changes the goals of an app's own model object. Run against a TEST Cale (scripts/cale_dev_server.py HOME), never the real one.

usage: python cale/tests/model_update.py [--base http://127.0.0.1:47899]
Nothing here belongs to one computer: sizes and speeds come from the machine under test.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cale.client import model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
a = ap.parse_args()
if ":47821" in a.base:
    sys.exit("This is the real Cale's port. Start a test Cale with scripts/cale_dev_server.py.")
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:300]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def call(path, body=None, key=None, method=None):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(a.base + path, data=None if body is None else json.dumps(body).encode(), headers=h, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=900)
        return r.status, dict(r.headers), json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read())


key = call("/cale/connect", {"app": "Update Test"})[2]["api_key"]
other = call("/cale/connect", {"app": "Other Update App"})[2]["api_key"]
with model(a.base + "/v1", key, name="chat", min_tokens_per_s=3, memory_pct=40, max_tokens=30) as m:
    s, _, r = call("/cale/models/chat", {"min_tokens_per_s": 9}, key)
    check("the goal changes", s == 200 and r["spec"]["min_tokens_per_s"] == 9.0, (s, r))
    check("the other goals stay", r["spec"].get("memory_pct") == 40.0 and r["spec"].get("max_tokens") == 30, r.get("spec"))
    check("the reply says what Cale predicts", "prediction" in r and "size_pct" in (r["prediction"].get("chosen") or {}), r.get("prediction"))
    s, _, r = call("/v1/models", key=key)
    mine = next(d for d in r["data"] if d["id"] == "chat")
    check("the list shows the new goal", mine["cale"]["spec"]["min_tokens_per_s"] == 9.0, mine)
    check("this computer reports its memory", (r["cale"]["this_computer"] or {}).get("ram_gb", 0) > 0, (r["cale"]["this_computer"] or {}).get("ram_gb"))
    sizes = (r["cale"]["this_computer"] or {}).get("sizes") or {}
    check("the speed of every size is listed", len(sizes) >= 2 and all("decode_tps" in v for v in sizes.values()), list(sizes)[:3])
    # a higher speed goal picks a smaller or equal size; a lower one a larger or equal size
    fast = max(v["decode_tps"] for v in sizes.values())
    slow = min(v["decode_tps"] for v in sizes.values())
    s, _, hi = call("/cale/models/chat", {"min_tokens_per_s": fast * 0.98}, key)
    s2, _, lo = call("/cale/models/chat", {"min_tokens_per_s": slow * 0.9}, key)
    check("a high speed goal picks a smaller size than a low one", hi["prediction"]["chosen"]["size_pct"] <= lo["prediction"]["chosen"]["size_pct"], (hi["prediction"].get("chosen"), lo["prediction"].get("chosen")))
    check("the lowest goal reaches a larger size than the highest", lo["prediction"]["chosen"]["size_pct"] > hi["prediction"]["chosen"]["size_pct"], (lo["prediction"]["chosen"]["size_pct"], hi["prediction"]["chosen"]["size_pct"]))
    # a real call uses the new goal
    call("/cale/models/chat", {"min_tokens_per_s": fast * 0.98}, key)
    s, hd, r = call("/v1/chat/completions", {"model": "chat", "max_tokens": 8, "messages": [{"role": "user", "content": "Say hi."}]}, key)
    size_fast = int(float(hd.get("X-Cale-Size", 0) or 0)) if s == 200 else None
    call("/cale/models/chat", {"min_tokens_per_s": slow * 0.9}, key)
    s, hd, r = call("/v1/chat/completions", {"model": "chat", "max_tokens": 8, "messages": [{"role": "user", "content": "Say hi."}]}, key)
    size_slow = int(float(hd.get("X-Cale-Size", 0) or 0)) if s == 200 else None
    check("calls run at the size the new goal asks for", s == 200 and size_fast is not None and size_slow is not None and size_fast <= size_slow, (size_fast, size_slow, hd.get("X-Cale-Size")))
    # refusals
    s, _, r = call("/cale/models/chat", {"memory_pct": 20}, key)
    check("a memory goal does not change here", s == 400, (s, r))
    s, _, r = call("/cale/models/chat", {"size": "fast"}, key)
    check("a size cannot be mixed in", s == 400, (s, r))
    s, _, r = call("/cale/models/chat", {"min_tokens_per_s": 9}, other)
    check("another app cannot change it", s == 404, (s, r))
    s, _, r = call("/cale/models/nothing", {"min_tokens_per_s": 9}, key)
    check("an unknown object is 404", s == 404, (s, r))
    s, _, r = call("/cale/models/chat", {"min_tokens_per_s": "fast"}, key)
    check("a bad value is 400", s == 400, (s, r))
    s, _, r = call("/cale/models/chat", {"min_tokens_per_s": None}, key)
    check("removing the only speed goal still leaves a goal (memory)", s == 200 and "min_tokens_per_s" not in r["spec"], (s, r))

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
