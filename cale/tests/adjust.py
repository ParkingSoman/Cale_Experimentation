"""The speed-goal adjuster of the batching engine (cale/batching.py _adjust_sustained). Run: python cale/tests/adjust.py   (no model, no engine)
A size must not change because of a slow moment that a smaller size does not fix (another app's prompt being read), and must not flip back and forth.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale import batching, gemma3n_facts as facts  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


LEVELS = facts.levels()


class Fake(batching.BatchMixin):
    def __init__(self, level):
        self.levels, self.batch_level = LEVELS, level
        self.mach = {"total_ram": 16e9}
        self.m = SimpleNamespace(missing=lambda l: False, set=lambda l: None)
        self.pred = SimpleNamespace(time_factor=lambda l: l["pct"] / 100.0)
        self.active = []
        self.machine = lambda: {}


def answer(goal_tps, pure_n, pure_s):
    r = batching.Request([], 16, 0.0, "a", 100, {"min_tokens_per_s": goal_tps}, 1.0, None, None, 0.0, False, None, None)
    r.goals = {"min_tokens_per_s": goal_tps, "min_quality": None, "memory_pct": None}
    r.fixed, r.max_pct = False, 100
    r.stream = SimpleNamespace(t_first=1.0, out=pure_n, switches=[])
    r.pure_n, r.pure_s = pure_n, pure_s
    return r


top = LEVELS[0]
e = Fake(top)
e.active = [answer(10.0, 5, 2.0)]                       # 2.5 tokens/s but only 2 s of evidence
e._adjust_sustained()
check("too little pure decoding: no change, whatever the speed", e.batch_level["id"] == top["id"])

e = Fake(top)
e.active = [answer(10.0, 20, 4.0)]                      # 5 tokens/s over 4 s: really slow
e._adjust_sustained()
check("sustained slow decoding: one size down", e.batch_level["id"] != top["id"] and e.batch_level["pct"] < top["pct"], e.batch_level["id"])
check("...and the counters start again", e.active[0].pure_n == 0 and e.active[0].pure_s == 0.0)
e._adjust_sustained()
check("...so no second step down right after", e.batch_level["pct"] == LEVELS[1]["pct"])

e = Fake(top)
e.active = [answer(10.0, 40, 4.0)]                      # 10 tokens/s: meets the goal
e._adjust_sustained()
check("meeting the goal: no change", e.batch_level["id"] == top["id"])

mid = LEVELS[2]
e = Fake(mid)
e.active = [answer(5.0, 70, 9.0)]                       # 7.8 tokens/s over 9 s: 1.5x the goal, three times STALL_S
e._adjust_sustained()
check("room to spare for a long time: one size up", e.batch_level["pct"] > mid["pct"], e.batch_level["id"])
e = Fake(mid)
e.active = [answer(5.0, 40, 5.0)]                       # 8 tokens/s over 5 s: fast, but not for long enough to go up
e._adjust_sustained()
check("room to spare for a short time: no change", e.batch_level["id"] == mid["id"])

e = Fake(top)
a, b = answer(10.0, 40, 4.0), answer(10.0, 20, 4.0)     # two apps: one fine, one slow
e.active = [a, b]
e._adjust_sustained()
check("one answer under its goal moves the step down", e.batch_level["pct"] < top["pct"])
e = Fake(top)
e.active = [answer(10.0, 20, 2.0), answer(10.0, 40, 4.0)]
e._adjust_sustained()
check("an answer with too little evidence holds the decision", e.batch_level["id"] == top["id"])

# the crowd a new answer will share steps with
e = Fake(top)
r_new = answer(10.0, 0, 0.0)
r_new.max_tokens = 100
short, long_, paused = answer(None, 0, 0.0), answer(None, 0, 0.0), answer(None, 0, 0.0)
short.max_tokens, long_.max_tokens, paused.max_tokens = 6, 400, 400
paused.paused = True
short.stream.out = long_.stream.out = 0
e.active = [short]
check("crowd: a 6-token answer next to a 100-token answer counts for 6 hundredths", abs(e._effective_crowd(r_new) - 1.06) < 1e-9, e._effective_crowd(r_new))
e.active = [long_]
check("crowd: a long answer counts in full", abs(e._effective_crowd(r_new) - 2.0) < 1e-9)
e.active = [paused]
check("crowd: a paused answer takes no steps", e._effective_crowd(r_new) == 1.0)
long_.stream.out = 380
e.active = [long_]
check("crowd: an answer that is nearly done counts for what is left", abs(e._effective_crowd(r_new) - 1.2) < 1e-9, e._effective_crowd(r_new))
e.active = []
check("crowd: alone is one", e._effective_crowd(r_new) == 1.0)

# the old and new rules are chosen by a setting
check("the rule can be chosen", batching.ADJUST_MODE in ("old", "sustained"))

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
