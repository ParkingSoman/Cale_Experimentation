"""Pick a size from goals. Pure Python: no MLX, no llama.cpp, so the server and the tests can import it anywhere.

`m` is what the engine knows about this computer's copy of the model (the *machine*): bytes of every layer's FFN weights,
the part that never changes, the disk speed. A *resident* list says, for each layer, which FFN state sits in RAM:
0 nothing, 1 half width, 2 full width (see `nest/gemma3n.py`). `pred` is a `predict.Predictor`.

| Function | Question it answers |
|---|---|
| need(level)            | which FFN state each layer needs to run this size |
| for_goals(...)         | the largest size that meets the goals (speed, first token, memory, quality), and what each size would do |
| for_memory(...)        | the largest size whose weights fit in a share of this computer's RAM (holds) |
No size, layer count or device name is written here: it works for any list of sizes.
"""
from . import gemma3n_facts as facts
from . import goals as goals_mod

SAFE_FRACTION = 0.85        # a setup that needs more than this share of the RAM is refused, whatever the app says
SAFE_MARGIN = 300_000_000   # and one that leaves less than this free right now


def need(level, n_layers=None):
    n_layers = len(level["blocks"]) if n_layers is None else n_layers
    return [0 if i in level["skip"] else level["blocks"][i] for i in range(n_layers)]


def union_need(levels, n_layers=None):
    return [max(col) for col in zip(*[need(l, n_layers) for l in levels])]


def ffn_bytes(m, states):
    return sum(m["layer_bytes"][i][s] for i, s in enumerate(states))


def model_bytes(m, states):
    """Weights in RAM for these layer states: the part that never changes plus the FFN part."""
    return m["base_bytes"] + ffn_bytes(m, states)


def disk_bytes(m, level, resident):
    """Bytes that must be read from disk to run `level` when `resident` is in RAM."""
    return sum(m["layer_read"][i] for i, n in enumerate(need(level, len(resident))) if n > resident[i])


def _largest_first(levels):
    return sorted(levels, key=lambda l: (-l["pct"], -l["mmlu_pt"]))


def memory_needed(m, level, context_tokens):
    """RAM this size needs to run a call: weights, the key/value cache of the context, and working memory for the prompt."""
    return model_bytes(m, need(level)) + facts.kv_bytes(context_tokens) + (m.get("scratch_bytes") or facts.SCRATCH_BYTES)


def row(level, pred, m, ctx, goals, resident):
    """What this size would do for this call, and which goals it meets."""
    d = disk_bytes(m, level, resident)
    disk_s = d / m["disk_bps"] if d else 0.0
    mem = memory_needed(m, level, ctx["prompt_tokens"] + ctx["answer_tokens"])
    held = max(0, model_bytes(m, resident) - 0) if resident else 0
    r = {"level": level, "decode_tps": pred.decode_tps(level, ctx["prompt_tokens"], ctx.get("answer_tokens")),
         "first_token_s": ctx.get("waited_s", 0.0) + pred.first_token_s(level, ctx["prompt_tokens"], disk_s),
         "memory_bytes": mem, "quality": facts.quality(level), "disk_bytes": d, "disk_s": disk_s}
    avail = ctx.get("available_ram")
    extra = max(0, mem - held)                                  # RAM this call adds to what the engine holds already
    r["safe"] = mem <= SAFE_FRACTION * ctx["total_ram"] and (avail is None or extra <= avail - SAFE_MARGIN)
    checks = {}
    if goals.get("min_tokens_per_s") is not None:
        checks[goals_mod.SPEED] = r["decode_tps"] >= goals["min_tokens_per_s"]
    if goals.get("max_first_token_s") is not None:
        checks[goals_mod.FIRST_TOKEN] = r["first_token_s"] <= goals["max_first_token_s"]
    if goals.get("memory_pct") is not None:
        checks[goals_mod.MEMORY] = mem <= ctx["total_ram"] * goals["memory_pct"] / 100
    if goals.get("min_quality") is not None:
        checks[goals_mod.QUALITY] = r["quality"] >= goals["min_quality"]
    r["checks"] = checks
    return r


def for_goals(levels, pred, m, goals, ctx, resident, cap_pct=100, light=False, allow_disk=False):
    """The largest size that meets the goals.

    ctx: {"prompt_tokens", "answer_tokens" (expected or maximum length), "waited_s", "total_ram", "available_ram" (or None)}.
    A size is left out if it exceeds cap_pct, is above the light-power ceiling (light=True), needs a disk read (unless allow_disk),
    or is not safe for this computer. If the goals conflict, `goals["priority"]` says which to keep: the largest size meeting the
    most important goals wins. Returns {"row", "met", "unmet", "rows", "unsafe"}. `unsafe` is True when no size is safe: then `row` is None.
    """
    pool = [l for l in levels if l["pct"] <= cap_pct + 0.5] or [min(levels, key=lambda l: l["pct"])]
    if light:
        pool = [l for l in pool if l["pct"] <= facts.LIGHT_MAX_PCT] or [min(pool, key=lambda l: l["pct"])]
    rows = [row(l, pred, m, ctx, goals, resident) for l in _largest_first(pool)]
    safe = [r for r in rows if r["safe"]]
    if not safe:
        return {"row": None, "met": False, "unmet": [], "rows": rows, "unsafe": True}
    ram = [r for r in safe if not r["disk_bytes"]]
    usable = safe if allow_disk else (ram or safe)              # nothing runs from RAM: the disk read is the only way
    active = [g for g in goals["priority"] if g in safe[0]["checks"]]
    for k in range(len(active), 0, -1):                         # keep the k most important goals, as many as possible
        keep = active[:k]
        ok = [r for r in usable if all(r["checks"][g] for g in keep)]
        if ok:
            best = max(ok, key=lambda r: (r["level"]["pct"], r["level"]["mmlu_pt"]))
            unmet = [g for g in active if not best["checks"][g]]
            return {"row": best, "met": not unmet, "unmet": unmet, "rows": rows, "unsafe": False}
    if active:                                                  # no size meets even the most important goal: the one that misses it least
        top = active[0]
        best = max(usable, key=lambda r: (_margin(top, r), r["level"]["pct"]))
        return {"row": best, "met": False, "unmet": [g for g in active if not best["checks"][g]], "rows": rows, "unsafe": False}
    best = max(usable, key=lambda r: (r["level"]["pct"], r["level"]["mmlu_pt"]))      # no goals at all
    return {"row": best, "met": True, "unmet": [], "rows": rows, "unsafe": False}


def _margin(goal, r):
    """How well this row does on one goal, higher is better. Used only to find the size that misses a goal least."""
    return {goals_mod.SPEED: r["decode_tps"], goals_mod.FIRST_TOKEN: -r["first_token_s"],
            goals_mod.MEMORY: -r["memory_bytes"], goals_mod.QUALITY: r["quality"]}[goal]


def for_memory(levels, m, total_ram, memory_pct, context_tokens=facts.HOLD_CONTEXT):
    """The largest size whose weights, plus the context and working room, fit in memory_pct of total_ram.

    If none fit, the smallest size is returned with met=False.
    Returns {"level", "bytes", "budget", "met"}.
    """
    budget = total_ram * memory_pct / 100
    rows = [{"level": l, "bytes": memory_needed(m, l, context_tokens)} for l in _largest_first(levels)]
    for r in rows:
        if r["bytes"] <= budget:
            return {**r, "budget": budget, "met": True}
    return {**min(rows, key=lambda r: r["bytes"]), "budget": budget, "met": False}
