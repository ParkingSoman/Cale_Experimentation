"""Passive samples (cale/samples.py, Engine._sample_call). Run: python cale/tests/samples.py   (no model, no server)
They must be cheap, never wait for the disk, hold numbers only, and trim themselves.
"""
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale import samples  # noqa: E402
from cale.engine import Engine  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:240]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def wait_for(cond, secs=5.0):
    t = time.time()
    while time.time() - t < secs:
        if cond():
            return True
        time.sleep(0.02)
    return False


tmp = Path(tempfile.mkdtemp(prefix="cale-samples-"))
s = samples.Samples(tmp / "samples.jsonl")
t0 = time.perf_counter()
for i in range(2000):
    s.record("call", tokens=i, tps=7.5 + i * 1e-4, where="cpu", hot=False)
cost_us = (time.perf_counter() - t0) / 2000 * 1e6
check("a sample costs the caller under 100 microseconds", cost_us < 100, f"{cost_us:.1f} us")
check("the lines reach the file", wait_for(lambda: len(s.read()) == 2000), len(s.read()))
check("the newest lines are also in memory", len(s.recent) == 200 and s.recent[-1]["tokens"] == 1999)
check("a line is short", max(len(l) for l in (tmp / "samples.jsonl").read_text().splitlines()) < 200)
bad = s.record("call", prompt_text=None, tokens=3)
check("a missing value is left out, not written as null", "prompt_text" not in bad and "null" not in json.dumps(bad))
s.path = tmp / "nope" / "samples.jsonl"            # a folder that is not there: the sample is lost, nothing else happens
check("a lost write does not raise", s.record("call", tokens=1) is not None)
time.sleep(0.2)

# trimming
tr = samples.Samples(tmp / "trim.jsonl")
(tmp / "trim.jsonl").write_text("\n".join(json.dumps({"kind": "call", "i": i}) for i in range(2 * samples.KEEP + 50)) + "\n")
tr._trim()
lines = (tmp / "trim.jsonl").read_text().splitlines()
check("a long file is cut to the newest KEEP lines", len(lines) == samples.KEEP and json.loads(lines[-1])["i"] == 2 * samples.KEEP + 49, len(lines))

# the engine's call sample
fake = SimpleNamespace(m=SimpleNamespace(placement="device=Vulkan1;gpu_layers=0"), samples=samples.Samples(tmp / "calls.jsonl"),
                       pred=SimpleNamespace(busy=lambda: 0.31, heat_now=lambda: 0.2))
done = {"size_pct": 48, "tier": "e2b", "tokens": 41, "decode_s": 14.0, "paused_dec_s": 4.0, "first_token_s": 2.5, "paused_pre_s": 0.5, "prompt_tokens": 300,
        "cached_tokens": 100, "batch_avg": 1.0, "waited_s": 0.0, "disk_s": 0.0, "step_ms": {"n": 40, "p50": 250.0, "p95": 300.0, "max": 900.0},
        "goal_met_actual": True, "finish": "stop"}
Engine._sample_call(fake, done)
got = fake.samples.recent[-1]
check("the call sample uses working time (paused time removed)", abs(got["tps"] - 40 / 10.0) < 0.01, got)
check("the call sample says where it ran, how busy the computer was, and the stall numbers",
      got["where"] == "gpu" and got["busy"] == 0.31 and got["step_max"] == 900.0 and got["step_p95"] == 300.0, got)
check("the call sample holds numbers and short words only", all(isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) < 12) for v in got.values()), got)
Engine._sample_call(SimpleNamespace(m=None, samples=samples.Samples(tmp / "x.jsonl"), pred=None), {"tokens": 1})
check("a call sample with nothing known does not raise", True)
sm = samples.step_summary([0.1, 0.2, 0.3, 2.0])
check("step summary: median, 95th, worst", sm["n"] == 4 and sm["max"] == 2000.0 and sm["p95"] == 2000.0 and sm["p50"] == 300.0, sm)
check("step summary of nothing is None", samples.step_summary([]) is None)

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
