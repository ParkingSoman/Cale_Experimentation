"""Beating a goal is a calibration error when a larger size was allowed (Engine._judge, demo/check_apps.py result).
Run: python cale/tests/calibration.py     (no model, no engine)
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cale import gemma3n_facts as facts, goals as G  # noqa: E402
from cale.engine import Engine  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


fake = SimpleNamespace(cap=1.0, levels=facts.levels(), mach={"total_ram": 16e9}, m=SimpleNamespace(active_bytes=lambda: 4e9))


def judge(goal, **done):
    d = {"tokens": 61, "decode_s": 6.0, "first_token_s": 0.5, "size_pct": 49, "start_tier": "x", "tier": "x", **done}
    Engine._judge(fake, d, G.parse(goal), 0.0)
    return d


d = judge({"min_tokens_per_s": 5})
check("twice the speed goal on a small size: wasted slice", d["goals_actual"]["speed"]["headroom"] == 2.0 and d["wasted_slice"] is True, d)
check("...and it still counts as met", d["goal_met_actual"] is True)
d = judge({"min_tokens_per_s": 5}, size_pct=100)
check("twice the goal on the largest size: nothing larger exists, not wasted", d["wasted_slice"] is False and d["headroom"] == 2.0, d)
d = judge({"min_tokens_per_s": 9.1}, size_pct=49)
check("10% over the goal is ordinary variation, not wasted", d["wasted_slice"] is False and 1.0 < d["headroom"] < 1.3, d)
d = judge({"min_tokens_per_s": 5, "max_first_token_s": 0.55})
check("a tight first-word goal holds the size down: the answer is not wasted", d["wasted_slice"] is False and d["headroom"] < 1.3, d)
d = judge({"min_tokens_per_s": 20})
check("a missed goal is not a wasted slice", d["goal_met_actual"] is False and d["wasted_slice"] is False and d["headroom"] < 1.0, d)
d = judge({"min_tokens_per_s": 5, "memory_pct": 25})
check("a memory goal counts too (4 GB used of a 4 GB budget leaves no room)", d["goals_actual"]["memory"]["headroom"] == 1.0 and d["wasted_slice"] is False, d)
d = judge({})
check("no goals: no headroom, no waste", d["headroom"] is None and d["wasted_slice"] is False, d)

sys.argv = ["check_apps"]
spec = importlib.util.spec_from_file_location("check_apps", ROOT / "demo" / "check_apps.py")
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)
hub = ca.summarize_calls([{"tokens_per_s": 10, "headroom": 2.0, "wasted_slice": True, "goal_met": True}] * 3 + [{"tokens_per_s": 6, "headroom": 1.1, "wasted_slice": False, "goal_met": True}])
check("the scorecard counts wasted slices", hub["wasted"] == 3 and hub["headroom"]["max"] == 2.0, hub)
r = ca.result("WELL", "fine", 1.0, hub=hub)
check("the scorecard calls mostly-wasted answers OVER, not WELL", r["verdict"] == "OVER" and "wasted slice" in r["notes"], r)
r = ca.result("WELL", "fine", 1.0, hub=ca.summarize_calls([{"headroom": 1.1, "wasted_slice": False, "goal_met": True}] * 4))
check("close to the goal stays WELL", r["verdict"] == "WELL", r)
r = ca.result("POOR", "slow", 1.0, hub=hub)
check("a POOR verdict is not changed", r["verdict"] == "POOR")

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
