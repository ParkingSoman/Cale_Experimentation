"""Measuring the ways of running again, in the background, when the saved choice is out of date (cale/engine.py replace_due, cale/tuning.py run_child).
Run: python cale/tests/replace.py     (no model, no server)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale import engine, tuning  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


NOW = 10 * 86400.0
base = dict(stale=True, replacing=False, idle_s=600.0, queued=False, on_battery=False, pressure_level=1, last_attempt=0.0, now=NOW)
due = lambda **kw: engine.replace_due(**{**base, **kw})          # noqa: E731
check("due: out of date, idle, plugged in, not tried today", due() is True)
check("not due: the saved choice still fits", due(stale=False) is False)
check("not due: Cale was used a minute ago", due(idle_s=60.0) is False)
check("not due: an app is waiting or running", due(queued=True) is False)
check("not due: on battery", due(on_battery=True) is False)
check("not due: short of memory", due(pressure_level=2) is False)
check("not due: already measuring", due(replacing=True) is False)
check("not due: tried less than a day ago", due(last_attempt=NOW - 3600) is False)
check("due again after a day", due(last_attempt=NOW - 90000) is True)
check("not due: Cale is being tuned", due(tuning=True) is False)

t = time.time()
r = tuning.run_child([sys.executable, "-c", "import time; time.sleep(30)"], None, timeout=60, stop=lambda: time.time() - t > 1.0, poll_s=0.2)
check("a running measurement stops within a second or two of the signal", r is None and time.time() - t < 6, time.time() - t)
r = tuning.run_child([sys.executable, "-c", "print('hello')"], None, timeout=30)
check("a finished child returns its output", r is not None and r[0] == 0 and r[1].strip() == "hello", r)
r = tuning.run_child([sys.executable, "-c", "import time; time.sleep(30)"], None, timeout=1, poll_s=0.2)
check("a child that takes too long is stopped", r is None)

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
