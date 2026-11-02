"""Cale learns how much a computer slows down on long work from real long answers (samples.pace, predict.Predictor.observe_pace).
Run: python cale/tests/pace.py   No model, no engine.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale.predict import Predictor  # noqa: E402
from cale.samples import pace  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


# --- samples.pace -----------------------------------------------------------------------------------------------------
steady = [0.03] * 400
check("a steady answer: early = late", abs(pace(steady)["early"] - pace(steady)["late"]) < 1e-6, pace(steady))
slowing = [0.03] * 200 + [0.036] * 200                                # 20% slower per step in the second half: speed ratio 0.833
p = pace(slowing)
check("a slowing answer: late is below early", p["late"] < p["early"], p)
check("the ratio is what the steps say", abs(p["late"] / p["early"] - 0.03 / 0.036) < 0.02, p["late"] / p["early"])
check("a short answer says nothing", pace([0.03] * 50) is None and pace([]) is None and pace(None) is None)
check("zero-time steps say nothing", pace([0.0] * 300) is None)
check("the first steps are left out (the engine is still settling)", abs(pace([1.0] * 4 + [0.03] * 300)["early"] - 1 / 0.03) < 0.5)

# --- Predictor.observe_pace --------------------------------------------------------------------------------------------
levels = [{"id": "e4b", "pct": 100, "eff_params_b": 3.98, "quality": 1.0}, {"id": "e2b", "pct": 48, "eff_params_b": 1.91, "quality": 0.8}]
tmp = Path(tempfile.mkdtemp(prefix="cale-pace-test-"))


def fresh(name="p.json"):
    return Predictor("llama", 20e9, {"engine": "llama"}, tmp / name, levels=levels)


hot = {"early": 10.0, "late": 8.0, "steps": 400}                        # 20% slower late
p1 = fresh()
check("nothing is known at the start", p1.sustained == 1.0 and p1.n_sus == 0)
check("the first long answer counts", p1.observe_pace(hot, 30.0, 0.0) is True)
check("...and sets the slowdown (nothing was known before)", abs(p1.sustained - 0.8) < 1e-9, p1.sustained)
check("a short answer is ignored", fresh().observe_pace(hot, 5.0, 0.0) is False)
check("an answer that started hot is ignored", fresh().observe_pace(hot, 30.0, 0.8) is False)
check("a debug 'slow' computer is ignored", (lambda q: (setattr(q, "slow", 2.0), q.observe_pace(hot, 30.0, 0.0))[1])(fresh()) is False)
noisy = {"early": 10.0, "late": 9.8, "steps": 400}
q = fresh(); q.observe_pace(noisy, 30.0, 0.0)
check("a small drop is noise: no throttling", q.sustained == 1.0 and q.n_sus == 1, (q.sustained, q.sust_raw))
check("the raw ratio is kept, so more answers can still add up", abs(q.sust_raw - 0.98) < 1e-9, q.sust_raw)
up = fresh(); up.observe_pace({"early": 10.0, "late": 14.0, "steps": 400}, 30.0, 0.0)
check("a ratio above 1 is held to 1.05 (noise), which means no throttling", up.sustained == 1.0 and up.sust_raw <= 1.05)
low = fresh(); low.observe_pace({"early": 10.0, "late": 2.0, "steps": 400}, 30.0, 0.0)
check("a ratio far below 0.5 is held at 0.5", abs(low.sustained - 0.5) < 1e-9, low.sustained)
# more answers move it slowly
q2 = fresh(); q2.observe_pace(hot, 30.0, 0.0)
q2.observe_pace({"early": 10.0, "late": 9.0, "steps": 400}, 30.0, 0.0)
check("a second answer moves it a quarter of the way", abs(q2.sust_raw - (0.75 * 0.8 + 0.25 * 0.9)) < 1e-9, q2.sust_raw)
# a value measured by `cale tune max` is the start
m = fresh(); m.extra = {"burst_s": 13}; m.sustained = 0.9; m.sust_raw = 0.9
m.observe_pace({"early": 10.0, "late": 8.0, "steps": 400}, 30.0, 0.0)
check("after a measured value, the first answer moves it half way", abs(m.sust_raw - (0.5 * 0.9 + 0.5 * 0.8)) < 1e-9, m.sust_raw)
# it is used in a prediction and it is saved
q3 = fresh("saved.json"); q3.table = {"e4b": {"dec": 30.0, "pre": 200.0}, "e2b": {"dec": 55.0, "pre": 400.0}}
before = q3.decode_tps(levels[0], 0, 3000)   # longer than the first 20 s of work (the burst before the slowing starts)
q3.observe_pace(hot, 30.0, 0.0)
after = q3.decode_tps(levels[0], 0, 3000)
short = q3.decode_tps(levels[0], 0, 100)
check("a long answer is predicted slower once the slowdown is learned", after < before, (before, after))
check("a short answer (inside the burst) is not", abs(short - before) < 1e-9 or short >= after, (short, before, after))
q3.save()
saved = json.loads((tmp / "saved.json").read_text())
check("the learned slowdown is saved", abs(saved["sustained"] - 0.8) < 1e-9 and saved["n_sus"] == 1, saved.get("sustained"))
q4 = Predictor("llama", 20e9, {"engine": "llama"}, tmp / "saved.json", levels=levels)
check("a new start loads it", abs(q4.sustained - 0.8) < 1e-9 and q4.n_sus == 1 and abs(q4.sust_raw - 0.8) < 1e-9, (q4.sustained, q4.n_sus))

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
