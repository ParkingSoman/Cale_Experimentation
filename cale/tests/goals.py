"""Goals against a running Cale (start one with scripts/cale_dev_server.py HOME): ask before you run, quality, priority,
the safety limit, light power, no hidden answer length, and the closed loop that changes size between tokens.
No number here belongs to one computer: speeds and sizes are read from the machine under test.

usage: python cale/tests/goals.py [--base http://127.0.0.1:47899]
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cale import gemma3n_facts as facts  # noqa: E402
from cale.client import CaleError, predict  # noqa: E402
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


def chat(key, prompt="Name the capital of France in one sentence.", model="gemma-3n:auto", **kw):
    return call("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}], **kw}, key)


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def tps(r):
    c, u = r["cale"], r["usage"]
    return (u["completion_tokens"] - 1) / max(c["seconds"] - c["first_token_s"], 1e-6)


key = call("/cale/connect", {"app": "Goals Test"})[2]["api_key"]
base = a.base + "/v1"

# 1. ask before you run
pr = predict(base, key, min_tokens_per_s=0.1, prompt_tokens=100)
sizes = pr["sizes"]
full_id = next(k for k, v in sizes.items() if v["size_pct"] == MAX_PCT)
check("predict: says which size Cale would use and what every size does", pr["goal_met"] and pr["chosen"]["size"] == full_id and len(sizes) >= 10, pr.get("chosen"))
check("predict: reports how sure it is", pr["confidence"] in ("estimate", "learned"), pr.get("confidence"))
check("predict: knows the floor of this computer", pr["floor"]["fastest"]["tokens_per_s"] > 0 and pr["floor"]["smallest"]["needs_gb"] > 0, pr.get("floor"))
pr2 = predict(base, key, min_tokens_per_s=10000)
check("predict: an impossible goal is reported, not raised", pr2["goal_met"] is False and pr2["unmet"] == ["speed"], pr2.get("unmet"))
pr3 = predict(base, key, min_tokens_per_s=10000, on_miss="refuse")
check("predict never refuses, even with on_miss=refuse", pr3["goal_met"] is False)
try:
    predict(base, key, deadline_s=5)
    check("predict refuses the removed deadline_s", False)
except CaleError as e:
    check("predict refuses the removed deadline_s", e.status == 400 and "unsupported_parameter" in e.body, e.body)
s, _, r = call("/cale/predict", {"min_tokens_per_s": 5, "bogus": 1}, key)
check("predict refuses an unknown field", s == 400 and r["error"]["code"] == "unknown_parameter", r)
m = call("/v1/models", key=key)[2]["cale"]
check("/v1/models: this computer, with a floor and a way to ask", m["this_computer"]["floor"] and "min_tokens_per_s" in m["sizes"]["goals"], m["sizes"].get("goals"))

# 2. quality
q_goal = 0.95
s, hd, r = chat(key, max_tokens=12, cale_min_quality=q_goal)
chosen = next(v for v in sizes.values() if v["size_pct"] == int(hd["X-Cale-Size"]))
check("min_quality: the size used repeats at least that share of the full size's choices", s == 200 and chosen["quality"] >= q_goal and hd["X-Cale-Goal-Met"] == "true", (dict(hd), chosen))

# 3. priority when goals conflict: a speed just above the full size's, and a quality only the full size has
speed = round(sizes[full_id]["tokens_per_s"] * 1.15, 2)
only_full = max(v["quality"] for k, v in sizes.items() if k != full_id) + 0.001
s, hd_q, r = chat(key, max_tokens=12, cale_min_tokens_per_s=speed, cale_min_quality=only_full, cale_priority=["quality"])
check("priority quality: keeps quality (the full size), reports speed unmet", s == 200 and hd_q["X-Cale-Size"] == str(MAX_PCT) and hd_q["X-Cale-Unmet"] == "speed", dict(hd_q))
s, hd_s, r = chat(key, max_tokens=12, cale_min_tokens_per_s=speed, cale_min_quality=only_full, cale_priority=["speed"])
check("priority speed: keeps speed (a smaller size), reports quality unmet", s == 200 and int(hd_s["X-Cale-Size"]) < MAX_PCT and hd_s["X-Cale-Unmet"] == "quality", dict(hd_s))
s, hd, r = chat(key, max_tokens=12, cale_min_tokens_per_s=speed, cale_min_quality=only_full, cale_priority=["speed", "bogus"])
check("bad priority: 400", s == 400 and r["error"]["code"] == "invalid_priority", r)

# 4. on_miss
s, hd, r = chat(key, max_tokens=12, cale_min_tokens_per_s=10000, cale_on_miss="try")
check("on_miss try: runs anyway and says the goal was not met", s == 200 and hd["X-Cale-Goal-Met"] == "false", dict(hd))
s, hd, r = chat(key, max_tokens=12, cale_min_tokens_per_s=10000, cale_on_miss="refuse")
e = r["error"]["cale"]
check("on_miss refuse: 422 says what is unmet, what every size would do, and the floor", s == 422 and e["unmet"] == ["speed"] and e["sizes"] and e["floor"], r)
s, hd, r = chat(key, max_tokens=12, cale_min_tokens_per_s=1, cale_on_miss="never")
check("bad on_miss: 400", s == 400 and r["error"]["code"] == "invalid_on_miss", r)

# 5. the safety limit: no setting overrides it, and it is a documented error
call("/cale/debug/ram", {"gb": 0.5})
for om in ("best_effort", "try", "refuse"):
    s, hd, r = chat(key, max_tokens=12, cale_min_tokens_per_s=1, cale_on_miss=om)
    check(f"unsafe for this computer, on_miss={om}: 507 with the numbers", s == 507 and r["error"]["code"] == "unsafe_for_this_computer" and "needed_gb" in r["error"]["cale"], (s, r))
s, hd, r = chat(key, model="gemma-3n:smart", max_tokens=12)
check("unsafe also stops an exact size", s == 507, (s, r))
pr4 = predict(base, key, min_tokens_per_s=1)
check("predict says unsafe too", pr4["unsafe"] is True and pr4["chosen"] is None, pr4)
call("/cale/debug/ram", {})
s, hd, r = chat(key, max_tokens=8, cale_min_tokens_per_s=0.2)
check("after the pretend is off: runs again", s == 200, (s, r))

# 6. power
s, hd, r = chat(key, max_tokens=8, cale_power="light")
check("power light: accepted; a smaller size only on battery or when hot", s == 200 and (int(hd["X-Cale-Size"]) <= facts.LIGHT_MAX_PCT or hd["X-Cale-Size"] == str(MAX_PCT)), dict(hd))
s, hd, r = chat(key, max_tokens=8, cale_power="turbo")
check("bad power: 400", s == 400 and r["error"]["code"] == "invalid_power", r)

# 7. no hidden answer length
s, hd, r = chat(key, "Say hi.")
check("no max_tokens: the answer is limited by the context, not by a hidden number", s == 200 and r["cale"]["answer_limit"] > 1024, r["cale"].get("answer_limit"))

# 8. first-token goal is judged after the answer too
s, hd, r = chat(key, max_tokens=8, cale_max_first_token_s=3600)
ga = r["cale"]["goals_actual"]["first_token"]
check("first-token goal: reply gives goal, actual and met", ga["met"] is True and ga["actual"] > 0, ga)

# 9. the closed loop: predictions that are too hopeful make Cale start big, then step down while the answer runs
s, hd, r = chat(key, "Write a long poem about the sea.", model=f"gemma-3n:{MAX_PCT}", max_tokens=40)
real_full = tps(r)
goal = round(real_full * 1.5, 2)
call("/cale/debug/slow", {"factor": 0.4})                   # the predictor now believes the computer is 2.5x faster than it is
s, hd, r = chat(key, "Write a long poem about the sea.", max_tokens=150, cale_min_tokens_per_s=goal)
call("/cale/debug/slow", {"factor": 1.0})
c = r["cale"]
check("closed loop: started at a size the (too hopeful) prediction liked", s == 200 and hd["X-Cale-Size"] == str(MAX_PCT), dict(hd))
check("closed loop: stepped down while answering, and says so", bool(c.get("size_changes")) and all(x["to"] != x["from"] for x in c["size_changes"]), c.get("size_changes"))
check("closed loop: ended at a smaller size than it started", r["model"] != f"gemma-3n:{MAX_PCT}", r["model"])
check("closed loop: the reply still says what was really achieved", c["goals_actual"]["speed"]["actual"] > 0, c["goals_actual"])

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
