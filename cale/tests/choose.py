"""Unit tests for the goals, the predictor and the size chooser (cale/goals.py, cale/predict.py, cale/choose.py).
Run: python cale/tests/choose.py   (from the repo folder; no model, no server)
Every number here is made up: nothing belongs to one computer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale import choose, gemma3n_facts as facts, goals as G  # noqa: E402
from cale.predict import Predictor  # noqa: E402

LEVELS = facts.levels()
N = 35
FULL, SMALL = LEVELS[0], LEVELS[-1]
GB = 1e9
M = {"base_bytes": 1.2 * GB, "disk_bps": 2 * GB, "total_ram": 24 * GB,
     "layer_bytes": [[0, 0.028 * GB, 0.066 * GB]] * N, "layer_read": [0.0566 * GB] * N}
ALL = [2] * N
SMALLEST = choose.need(SMALL)
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


class Pred:
    """A computer whose full size answers at 30 tokens/s and reads a prompt at 600 tokens/s. Sizes scale with their cost."""
    confidence = "learned"

    def __init__(self, slow=1.0):
        self.slow = slow

    def decode_tps(self, l, ctx=0, answer_tokens=None):
        return 30 / self.slow / (0.1 + 0.9 * facts.cost(l))

    def prefill_tps(self, l, tokens=None):
        return 600 / self.slow / facts.cost(l)

    def first_token_s(self, l, prompt_tokens, disk_s=0.0):
        return disk_s + prompt_tokens / self.prefill_tps(l) + 1 / self.decode_tps(l)


CTX = {"prompt_tokens": 300, "answer_tokens": 100, "waited_s": 0.0, "total_ram": 24 * GB, "available_ram": 20 * GB}


def goals(**kw):
    return G.parse(kw)


def pick(g, pred=None, ctx=None, resident=ALL, **kw):
    return choose.for_goals(LEVELS, pred or Pred(), M, g, ctx or CTX, resident, **kw)


# --- sizes and memory
check("need: full size needs every layer full", choose.need(FULL) == [2] * N)
check("need: the smallest size skips 5 layers", choose.need(SMALL).count(0) == 5)
check("union of sizes", choose.union_need([SMALL, FULL]) == [2] * N and choose.union_need([SMALL]) == SMALLEST)
check("disk bytes: nothing when resident", choose.disk_bytes(M, FULL, ALL) == 0 and choose.disk_bytes(M, SMALL, SMALLEST) == 0)
check("disk bytes: counted for layers that grow", choose.disk_bytes(M, FULL, SMALLEST) == M["layer_read"][0] * sum(1 for a, b in zip(choose.need(FULL), SMALLEST) if a > b))
r = choose.for_memory(LEVELS, M, 24 * GB, 50)
check("memory goal: roomy goal gives the full size", r["level"]["id"] == FULL["id"] and r["met"])
small = choose.memory_needed(M, SMALL, facts.HOLD_CONTEXT)
r = choose.for_memory(LEVELS, M, 24 * GB, 100 * small / (24 * GB))
check("memory goal: exactly the smallest size's need gives the smallest size", r["met"] and r["level"]["id"] == SMALL["id"])
r = choose.for_memory(LEVELS, M, 24 * GB, 1)
check("memory goal: impossible gives the smallest, not met", not r["met"] and r["level"]["id"] == SMALL["id"])
big_ctx = choose.memory_needed(M, FULL, 32768) - choose.memory_needed(M, FULL, 100)
check("memory counts the context", abs(big_ctx - 32668 * facts.KV_BYTES_PER_TOKEN) < 1)

# --- the chooser
res = pick(goals())
check("no goals: the largest size that needs no disk", res["met"] and res["row"]["level"]["id"] == FULL["id"])
res = pick(goals(min_tokens_per_s=40))
check("speed goal: the largest size that keeps up", res["met"] and res["row"]["decode_tps"] >= 40 and res["row"]["level"]["pct"] < 100)
bigger = [r for r in res["rows"] if r["level"]["pct"] > res["row"]["level"]["pct"]]
check("...and every larger size is too slow", all(r["decode_tps"] < 40 for r in bigger))
res = pick(goals(min_tokens_per_s=20), Pred(slow=3.0))
fast = pick(goals(min_tokens_per_s=20))["row"]["level"]["pct"]
check("a 3x slower computer picks a smaller size", res["row"]["level"]["pct"] < fast)
res = pick(goals(min_tokens_per_s=9000))
check("impossible speed: the fastest size, not met, says speed", not res["met"] and res["unmet"] == ["speed"] and res["row"]["level"]["id"] == SMALL["id"])
res = pick(goals(max_first_token_s=0.8))
check("first-token goal is judged on prompt reading", res["met"] and res["row"]["first_token_s"] <= 0.8)
res = pick(goals(max_first_token_s=0.8), ctx={**CTX, "waited_s": 0.79})
check("time waited in line counts toward the first token", res["row"]["first_token_s"] > 0.79 and not res["met"] or res["row"]["level"]["pct"] < 100)
res = pick(goals(min_quality=0.9))
check("quality floor: no size below it", res["met"] and res["row"]["quality"] >= 0.9)
res = pick(goals(memory_pct=15))
check("memory goal on a call: the size fits the share", res["met"] and res["row"]["memory_bytes"] <= 0.15 * 24 * GB and res["row"]["level"]["pct"] < 100, res["row"]["memory_bytes"] / GB)
res = pick(goals(memory_pct=1))
check("impossible memory: the smallest, says memory", not res["met"] and res["unmet"] == ["memory"] and res["row"]["level"]["id"] == SMALL["id"])
res = pick(goals(), cap_pct=75)
check("cap (memory pressure): never above it", res["row"]["level"]["pct"] <= 75)
res = pick(goals(power="light"), light=True)
check("light power on battery: no size above the light ceiling", res["row"]["level"]["pct"] <= facts.LIGHT_MAX_PCT)
res = pick(goals(power="light"), light=False)
check("light power on mains: the full size", res["row"]["level"]["id"] == FULL["id"])

# --- conflicts and priority
g = goals(min_tokens_per_s=40, min_quality=0.95)
res = pick(g)
check("conflicting goals: some goal is reported unmet", not res["met"] and len(res["unmet"]) == 1, res["unmet"])
res_q = pick(goals(min_tokens_per_s=40, min_quality=0.95, priority=["quality"]))
res_s = pick(goals(min_tokens_per_s=40, min_quality=0.95, priority=["speed"]))
check("priority quality: keeps quality, gives up speed", res_q["unmet"] == ["speed"] and res_q["row"]["quality"] >= 0.95)
check("priority speed: keeps speed, gives up quality", res_s["unmet"] == ["quality"] and res_s["row"]["decode_tps"] >= 40)

# --- disk
res = pick(goals(min_tokens_per_s=1), resident=SMALLEST)
check("no disk permission: stays on what is in RAM", res["row"]["level"]["id"] == SMALL["id"] and res["row"]["disk_bytes"] == 0)
res = pick(goals(min_tokens_per_s=1), resident=SMALLEST, allow_disk=True)
check("disk permission: the larger size, with the read counted", res["row"]["level"]["id"] == FULL["id"] and res["row"]["disk_bytes"] > 0 and res["row"]["first_token_s"] > res["row"]["disk_s"])
slow_disk = dict(M, disk_bps=1e6)
res = choose.for_goals(LEVELS, Pred(), slow_disk, goals(max_first_token_s=5), CTX, SMALLEST, allow_disk=True)
check("a slow disk makes the read-from-disk size miss a first-token goal", res["row"]["level"]["id"] != FULL["id"])

# --- the safety limit
res = choose.for_goals(LEVELS, Pred(), M, goals(), {**CTX, "total_ram": 4 * GB}, ALL)
check("safety: a size that needs most of the RAM is not chosen", not res["unsafe"] and res["row"]["level"]["id"] != FULL["id"] and all(r["safe"] for r in [res["row"]]))
res = choose.for_goals(LEVELS, Pred(), M, goals(), {**CTX, "total_ram": 1 * GB}, ALL)
check("safety: when no size is safe, nothing is chosen", res["unsafe"] and res["row"] is None)
res = choose.for_goals(LEVELS, Pred(), M, goals(), {**CTX, "available_ram": 0.3 * GB}, SMALLEST)
check("safety: little free memory right now blocks growing", res["unsafe"] or res["row"]["memory_bytes"] <= choose.model_bytes(M, SMALLEST) + 0.3 * GB)

# --- goal parsing
def _refused(req):
    try:
        G.parse(req)
    except G.GoalError as e:
        return e.code


check("parse: defaults", G.parse({})["on_miss"] == "best_effort" and G.parse({})["priority"] == G.DEFAULT_PRIORITY and not G.is_set(G.parse({})))
check("parse: old on_miss 'smallest' is best_effort", G.parse({"on_miss": "smallest"})["on_miss"] == "best_effort")
check("parse: max_wait_s is capped (the generation lock is held while waiting)", _refused({"max_wait_s": 3600}) == "invalid_max_wait_s")
check("parse: prefix", G.parse({"cale_min_tokens_per_s": 5}, "cale_")["min_tokens_per_s"] == 5.0)
check("parse: a partial priority list is completed", G.parse({"priority": ["speed"]})["priority"][0] == "speed" and len(G.parse({"priority": ["speed"]})["priority"]) == 4)
for bad, code in (({"deadline_s": 3}, "unsupported_parameter"), ({"min_tokens_per_s": 0}, "invalid_min_tokens_per_s"), ({"min_tokens_per_s": True}, "invalid_min_tokens_per_s"),
                  ({"memory_pct": 101}, "invalid_memory_pct"), ({"min_quality": 1.5}, "invalid_min_quality"), ({"power": "turbo"}, "invalid_power"),
                  ({"on_miss": "x"}, "invalid_on_miss"), ({"priority": ["speed", "speed"]}, "invalid_priority"), ({"allow_disk": "yes"}, "invalid_allow_disk")):
    try:
        G.parse(bad)
        check(f"parse refuses {bad}", False)
    except G.GoalError as e:
        check(f"parse refuses {bad}", e.code == code, e.code)

# --- the predictor
p = Predictor("llama", 28e9, levels=LEVELS)
check("predictor: starts as an estimate, no model timing", p.confidence == "estimate" and p.decode_tps(FULL) > 0)
check("predictor: smaller sizes are faster", all(p.decode_tps(a) <= p.decode_tps(b) for a, b in zip(LEVELS, LEVELS[1:])))
p.observe({"tier": FULL["id"], "tokens": 40, "decode_s": 39 / 8.0, "prompt_tokens": 200, "first_token_s": 200 / 20 + 0.125})
check("predictor: one real answer sets the full size's speed within 15%", abs(p.decode_tps(FULL) - 8.0) / 8.0 < 0.15 and p.confidence == "learned", p.decode_tps(FULL))
check("predictor: ...and the prompt speed", abs(p.prefill_tps(FULL) - 20) / 20 < 0.2, p.prefill_tps(FULL))
pp = Predictor("llama", 28e9, levels=LEVELS)       # an answer held back for another app: its clock ran, but it was not working
pp.observe({"tier": FULL["id"], "tokens": 40, "decode_s": 39 / 8.0 + 30.0, "paused_dec_s": 30.0, "prompt_tokens": 200,
            "first_token_s": 200 / 20 + 0.125 + 45.0, "paused_pre_s": 45.0})
check("predictor: time an answer spent paused is not learned as slow decode", abs(pp.decode_tps(FULL) - 8.0) / 8.0 < 0.15, pp.decode_tps(FULL))
check("predictor: ...nor as slow prompt reading", abs(pp.prefill_tps(FULL) - 20) / 20 < 0.2, pp.prefill_tps(FULL))
pn = Predictor("llama", 28e9, levels=LEVELS)       # the same answer without the pause fields shows what the bug did
pn.observe({"tier": FULL["id"], "tokens": 40, "decode_s": 39 / 8.0 + 30.0, "prompt_tokens": 200, "first_token_s": 200 / 20 + 0.125 + 45.0})
check("predictor: without the pause fields the same answer looks much slower (the old behavior)", pn.decode_tps(FULL) < 0.5 * pp.decode_tps(FULL), pn.decode_tps(FULL))
before = p.decode_tps(FULL)
p.observe({"tier": FULL["id"], "tokens": 40, "decode_s": 39 / 4.0, "prompt_tokens": 200, "first_token_s": 20, "switches": [{"at_token": 16}]})
check("predictor: an answer that changed size is not used for speed", p.decode_tps(FULL) == before)
p.slow = 2.0
check("predictor: debug slowdown halves the speed", abs(p.decode_tps(FULL) - before / 2) < 1e-6)
p.slow = 1.0
p.busy = lambda: 1.0
check("predictor: a busy computer predicts slower", p.decode_tps(FULL) < before)
# --- what other programs cost (the load factor learns from answers)
pl = Predictor("llama", 28e9, levels=LEVELS)
check("load: the prior says half load costs about a third, full load a half", abs(1 / (1 + pl.load_alpha * 0.5) - 0.667) < 0.01 and abs(1 / (1 + pl.load_alpha) - 0.5) < 0.01)
for _ in range(3):
    pl.busy = lambda: 0.02
    pl.observe({"tier": FULL["id"], "tokens": 40, "decode_s": 39 / 8.0, "prompt_tokens": 100, "first_token_s": 5})
pl.busy = lambda: 0.5
pl.observe({"tier": FULL["id"], "tokens": 40, "decode_s": 39 / 3.2, "prompt_tokens": 100, "first_token_s": 5})        # 8 -> 3.2 tok/s at half load: alpha 3
check("load: a computer that slows more than the prior says teaches it", pl.load_alpha > 1.5 and pl.n_load == 1, pl.load_alpha)
for _ in range(6):
    pl.observe({"tier": FULL["id"], "tokens": 40, "decode_s": 39 / 3.2, "prompt_tokens": 100, "first_token_s": 5})
check("load: ...and it settles near what the answers show (alpha about 3)", 2.2 < pl.load_alpha < 3.4, pl.load_alpha)
check("load: the learned factor is used in the prediction", abs(pl.live_factor() - 1 / (1 + pl.load_alpha * 0.5)) < 0.02, pl.live_factor())
pc = Predictor("llama", 28e9, levels=LEVELS)
pc.busy = lambda: 0.5
pc.observe({"tier": FULL["id"], "tokens": 40, "decode_s": 39 / 3.2, "prompt_tokens": 100, "first_token_s": 5})
check("load: with no quiet answers yet nothing is learned (no baseline)", pc.n_load == 0 and pc.load_alpha == 1.0)
pc2 = Predictor("llama", 28e9, levels=LEVELS)
pc2.busy = lambda: 0.02
for _ in range(3):
    pc2.observe({"tier": FULL["id"], "tokens": 40, "decode_s": 39 / 8.0, "prompt_tokens": 100, "first_token_s": 5})
pc2.busy = lambda: 0.5
pc2.observe({"tier": FULL["id"], "tokens": 40, "decode_s": 39 / 3.2, "prompt_tokens": 100, "first_token_s": 5, "batch_avg": 2.0})
check("load: an answer that shared steps with another answer teaches nothing about other programs", pc2.n_load == 0)

# --- interruptions: learn the speed of decoding alone, and the cost of being interrupted as a separate number
ps = Predictor("llama", 28e9, levels=LEVELS)
for _ in range(30):
    ps.observe({"tier": FULL["id"], "tokens": 41, "decode_s": 40 / 5.5, "pure_tps": 8.0, "prompt_tokens": 100, "first_token_s": 5})        # felt 5.5, decoding alone 8.0
check("steal: decode speed is learned from decoding alone (8, not the felt 5.5)", abs(ps._decode_pure(FULL) - 8.0) / 8.0 < 0.1, ps._decode_pure(FULL))
check("steal: the interruption cost is learned (about 31%)", 0.25 < ps.steal < 0.37, ps.steal)
check("steal: the speed the app feels is decode-only less the steal (about 5.5)", abs(ps.decode_tps(FULL) - 5.5) / 5.5 < 0.12, ps.decode_tps(FULL))
pn2 = Predictor("llama", 28e9, levels=LEVELS)
for _ in range(8):
    pn2.observe({"tier": FULL["id"], "tokens": 41, "decode_s": 40 / 5.5, "prompt_tokens": 100, "first_token_s": 5})                      # no pure speed known: the old behavior
check("steal: without a decode-only speed nothing is split (the old behavior)", pn2.steal == 0.0 and abs(pn2.decode_tps(FULL) - 5.5) / 5.5 < 0.12)

# --- smoothing of other programs' CPU use
from cale.predict import Smooth  # noqa: E402
sm = Smooth(6.0)
sm.update(1.0, 2.0)
check("smooth: one spike of a 2 s reading moves the value by a third at most", sm.value < 0.34, sm.value)
for _ in range(10):
    sm.update(0.0, 2.0)
check("smooth: it settles back", sm.value < 0.02, sm.value)
for _ in range(15):
    sm.update(0.5, 2.0)
check("smooth: a lasting load shows in full", abs(sm.value - 0.5) < 0.02, sm.value)

# --- the install-time table must not say a smaller size is slower
from cale import tuning  # noqa: E402
tbl = {"big": {"pre": 24.0, "dec": 5.6}, "mid": {"pre": 30.0, "dec": 7.7}, "bad": {"pre": 40.0, "dec": 5.1}, "small": {"pre": 42.0, "dec": 12.6}}
redo = {"bad": [(41.0, 8.4)]}
fixed = tuning.repair_monotone(["big", "mid", "bad", "small"], tbl, lambda i: redo[i].pop(0) if redo.get(i) else None)
check("table: a smaller size measured slower is measured again", fixed == ["bad"], fixed)
check("table: the better reading is kept (8.4, not 5.1)", tbl["bad"]["dec"] == 8.4 and tbl["bad"]["pre"] == 41.0, tbl["bad"])
tbl2 = {"big": {"pre": 24.0, "dec": 5.6}, "small": {"pre": 42.0, "dec": 12.6}}
check("table: a sound table is left alone (no extra runs)", tuning.repair_monotone(["big", "small"], tbl2, lambda i: (_ for _ in ()).throw(AssertionError("no extra run needed"))) == [])
tbl3 = {"big": {"pre": 24.0, "dec": 9.0}, "small": {"pre": 42.0, "dec": 5.0}}
n3 = []
fixed3 = tuning.repair_monotone(["big", "small"], tbl3, lambda i: (n3.append(i), (40.0, 5.5))[1], tries=2)
check("table: it does not loop forever when the order stays wrong", len(n3) == 2 and tbl3["small"]["dec"] == 5.5, n3)

p2 = Predictor("mlx", 100e9, levels=LEVELS)
check("predictor: a faster memory predicts faster", p2.decode_tps(FULL) > Predictor("llama", 28e9, levels=LEVELS).decode_tps(FULL))

# the device rule (cale/backends/devices.py): no device name and no layer count in it
from cale.backends import devices as dv  # noqa: E402
big = {"handle": 0, "type": "GPU", "name": "Big", "description": "", "free": 24 << 30, "total": 24 << 30}
small = {"handle": 1, "type": "GPU", "name": "Small", "description": "", "free": 2 << 30, "total": 2 << 30}
igpu = {"handle": 2, "type": "IGPU", "name": "Shared", "description": "", "free": 8 << 30, "total": 8 << 30}
model = 4 << 30
check("devices: a GPU that holds the model gets every layer", dv.choose([small, big], model, "auto")[:2] == (big, 999))
check("devices: a small GPU gets no layers when the library can read weights back", dv.choose([small], model, "auto", readback=True)[:2] == (small, 0))
check("devices: a small GPU is not used when the library cannot", dv.choose([small], model, "auto", readback=False)[0] is None)
check("devices: a GPU that shares memory is never chosen on its own", dv.choose([igpu], model, "auto", readback=True)[0] is None)
check("devices: llama.cpp says every layer fits -> every layer, even if the rough rule says no", dv.choose([small], model, "auto", readback=True, fits=lambda d: True)[:2] == (small, 999))
check("devices: llama.cpp says not every layer fits -> reads prompts, even if the rough rule says yes", dv.choose([big], model, "auto", readback=True, fits=lambda d: False)[:2] == (big, 0))
check("devices: llama.cpp cannot tell -> the rough rule", dv.choose([big], model, "auto", fits=lambda d: None)[:2] == (big, 999))
cu = {"handle": 3, "type": "GPU", "name": "CUDA0", "description": "NVIDIA Test", "free": 2 << 30, "total": 2 << 30}
vk = {"handle": 4, "type": "GPU", "name": "Vulkan1", "description": "NVIDIA Test", "free": 2 << 30, "total": 2 << 30}
check("devices: a card seen by CUDA and Vulkan is used through CUDA", dv.choose([vk, cu], model, "auto", readback=True)[0] is cu and dv.choose([cu, vk], model, "auto", readback=True)[0] is cu)
check("devices: Vulkan alone is still used", dv.choose([vk], model, "auto", readback=True)[0] is vk)
check("devices: off means the CPU", dv.choose([big], model, "off", readback=True)[0] is None)
check("devices: partial asks for llama.cpp's fit", dv.choose([small], model, "partial")[:2] == (small, -1))

sv = {"use": "Small", "mode": "full"}
check("devices: a saved 'full' is checked against free memory now", dv.choose([small], model, "auto", readback=True, fits=lambda d: False, saved=sv)[:2] == (small, 0))
check("devices: a saved 'full' without readback falls to the CPU when memory is not free", dv.choose([small], model, "auto", readback=False, fits=lambda d: False, saved=sv)[0] is None)
check("devices: a saved 'full' is kept when memory is free", dv.choose([small], model, "auto", readback=True, fits=lambda d: True, saved=sv)[:2] == (small, 999))
check("devices: a saved device that is gone is said in the reason", "not here now" in dv.choose([big], model, "auto", readback=True, saved={"use": "Gone", "mode": "prompts"})[2])

# tuning: a probe runs in a process with a temporary home, so it must be told where the model and the library are
import subprocess
from cale import tuning
seen = {}
class _R:
    stdout = 'PLACEMENT_PROBE {"pre": 1.0, "dec": 1.0}'; stderr = ""; returncode = 0
_real = tuning.run_child
tuning.run_child = lambda cmd, env, timeout=900, stop=None, poll_s=0.5: (seen.update(env), (0, _R.stdout, ""))[1]
tuning.PROBE_ENV.update({"CALE_MODEL": "/m/model.gguf", "CALE_LLAMA_LIB": "/l/lib.so"})
tuning.probe_one("off", lambda *a: None)
tuning.run_child = _real
check("tune: a probe process is told the model and the library", seen.get("CALE_MODEL") == "/m/model.gguf" and seen.get("CALE_LLAMA_LIB") == "/l/lib.so" and seen.get("CALE_GPU") == "off")

print("\nFAILED: " + ", ".join(FAIL) if FAIL else "\nall passed")
sys.exit(1 if FAIL else 0)
