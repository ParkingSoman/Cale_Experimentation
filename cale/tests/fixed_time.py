"""The fixed-time readings of the hardware test (cale/tuning.py Tuner.timed and Tuner.point): when a window ends, how the speed is read from it, and what happens
to a window with a stall in it. Run: python cale/tests/fixed_time.py   No model: the engine is a stand-in that reports made-up times.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale import tuning  # noqa: E402
from nest.cancel import Cancelled  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


class FakeModel:
    """Answers at made-up times. `script(n)` gives the time (seconds since the first token) at which token n arrives. Calls the controller every 8 tokens, like the engines."""

    def __init__(self, script):
        self.script, self.calls, self.cancel_set = script, 0, False

    def prompt_len(self, msgs):
        return 48

    def clear_cache(self):
        pass

    def chat_stream(self, msgs, max_tokens=256, width=None, temperature=0.0, cancel=None, controller=None, **kw):
        self.calls += 1
        for n in range(1, max_tokens + 1):
            if cancel is not None and cancel.is_set():
                self.cancel_set = True
                raise Cancelled()
            yield {"delta": "x"}
            if controller and n % 8 == 0:
                controller(n, self.script(n), None)


def tuner_for(script):
    level = {"id": "full", "pct": 100, "eff_params_b": 4.0}
    eng = SimpleNamespace(m=FakeModel(script), levels=[level], pred=SimpleNamespace(table={}, table_prompt=48))
    t = tuning.Tuner(eng, "quick")
    t.warmed = True
    return t, level


# 1. a fast computer: 40 tokens/s. The window ends after 2 s (and 32 tokens).
t, lv = tuner_for(lambda n: n / 40.0)
pre, dec = t.timed(lv)
check("fast: the speed is right", abs(dec - 40) < 1, dec)
check("fast: it was stopped (the window ends, the answer does not run on)", t.eng.m.cancel_set)
check("fast: it stopped after about 2 s (80 tokens), not at the 32-token floor", t.eng.m.calls == 1)

# 2. a slow computer: 5 tokens/s. The 32-token floor holds the window open past 2 s, but not past 7 s.
t, lv = tuner_for(lambda n: n / 5.0)
seen = []
orig = t.eng.m.chat_stream


def spy(*a, **k):
    ctl = k.get("controller")
    k["controller"] = lambda n, tt, l: (seen.append((n, round(tt, 2))), ctl(n, tt, l))[1]
    return orig(*a, **k)


t.eng.m.chat_stream = spy
pre, dec = t.timed(lv)
check("slow: the speed is right", abs(dec - 5) < 0.2, dec)
check("slow: it waited for 32 tokens (6.4 s), under the 7 s cap", seen[-1][0] == 32 and seen[-1][1] <= tuning.WINDOW_MAX_S, seen)

# 3. a very slow computer: 2 tokens/s. The cap ends it: fewer tokens, a less exact reading, never a longer test.
t, lv = tuner_for(lambda n: n / 2.0)
seen.clear()
orig = t.eng.m.chat_stream
t.eng.m.chat_stream = spy
pre, dec = t.timed(lv)
check("very slow: the cap ends the window (at the first report after 7 s)", seen[-1][1] >= tuning.WINDOW_MAX_S and seen[-1][0] < 32 and seen[-1][1] < tuning.WINDOW_MAX_S + 8, seen)
check("very slow: the speed is still read", abs(dec - 2) < 0.3, dec)

# 4. a stall in the middle: 7 tokens/s, but tokens 24 to 32 take 3 s longer. Noise only slows: the faster intervals tell the speed.
def stalled(n):
    return n / 7.0 + (3.0 if n >= 32 else 0.0)
t, lv = tuner_for(stalled)
pre, dec = t.timed(lv)
plain = (40 - 8) / (stalled(40) - stalled(8))
check("a stall does not drag the speed down", dec > 0.9 * 7 and dec > plain + 0.5, (dec, plain))
check("a window with a stall is marked unsteady", t.unsteady is True)

# 5. an unsteady window is taken again once, and the faster reading is kept
calls = []


class TwoWindows(FakeModel):
    """The first window uses scripts[0], later ones scripts[1]."""

    def __init__(self, scripts):
        super().__init__(scripts[0])
        self.scripts = scripts

    def chat_stream(self, *a, **k):
        self.script = self.scripts[min(len(calls), len(self.scripts) - 1)]
        calls.append(1)
        yield from super().chat_stream(*a, **k)


t, lv = tuner_for(stalled)
t.eng.m = TwoWindows([stalled, lambda n: n / 7.0])
pre, dec = t.point(lv, 48, repeats=1)
check("point: a window with a stall is repeated once", len(calls) == 2, len(calls))
check("point: the faster of the two readings is kept", abs(dec - 7) < 0.5, dec)
calls.clear()
t, lv = tuner_for(lambda n: n / 7.0)
t.eng.m = TwoWindows([lambda n: n / 7.0])
t.point(lv, 48, repeats=1)
check("point: a steady window is read once", len(calls) == 1, len(calls))

# 6. too little seen
t, lv = tuner_for(lambda n: n / 7.0)
t.eng.m = FakeModel(lambda n: n / 7.0)
t.eng.m.chat_stream = lambda *a, **k: iter(())
check("an answer that never starts gives no reading", t.timed(lv) is None)

# 7. the policy numbers hold together
check("policy: the floor can be reached inside the cap at 5 tokens/s", tuning.MIN_TOKENS / 5.0 <= tuning.WINDOW_MAX_S)
check("policy: the settling window is no longer than a reading", tuning.SETTLE_S <= tuning.WINDOW_S)

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
