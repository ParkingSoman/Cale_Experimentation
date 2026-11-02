"""Debug aid: run the fixed-time reading at different thread counts and print what the engine reported. usage: python scripts/debug_timed.py HOME"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.update(CALE_HOME=sys.argv[1], CALE_DEBUG="1", CALE_PORT="47899")
from cale import paths
from cale.engine import Engine
from cale.tuning import Tuner
from nest.cancel import Cancel, Cancelled

eng = Engine(paths.MODEL)
eng._tuning = True
with eng.lock:
    eng._load()
m = eng.m
print("placement:", getattr(m, "placement", None), "threads:", getattr(m, "threads", None))
t = Tuner(eng, "quick")
full = t.full_level()
t.warm_up()
print("warm speed", t.warm_speed, "table_prompt", t.pred.table_prompt)
if os.environ.get("DEBUG_MEMORY_FIRST") == "1":
    print("working_memory:", t.working_memory())
for n in (4, 8, 4, 8):
    m.set_threads(n, n)
    seen = []
    cancel = Cancel()
    unit = "The sea is wide and deep. "
    n_unit = max(1, int(m.prompt_len([{"role": "user", "content": unit}])))
    msgs = [{"role": "user", "content": unit * max(1, int(48 / max(n_unit - 8, 1))) + "Write the whole numbers from 1 to 300, separated by spaces."}]
    t0 = time.perf_counter(); first = None; n_deltas = 0

    def rep(k, tt, lvl):
        seen.append((k, round(tt, 2)))
        if (tt >= 2 and k >= 32) or tt >= 7:
            cancel.set()
    try:
        for ev in m.chat_stream(msgs, max_tokens=600, width=full, temperature=0.0, cancel=cancel, controller=rep):
            if "delta" in ev:
                n_deltas += 1
                if first is None:
                    first = round(time.perf_counter() - t0, 2)
    except Cancelled:
        print("  cancelled")
    m.clear_cache()
    print(f"threads {n}: first token at {first} s, deltas {n_deltas}, reports {seen}, total {time.perf_counter() - t0:.1f} s")
