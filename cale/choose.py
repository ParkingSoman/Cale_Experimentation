"""Pick a size from a goal. Pure Python: no MLX, so the server and the tests can import it anywhere.

A *profile* (see `cale/profile.py`) is what Cale measured on this Mac. A *resident* list says, for each
layer, which FFN state sits in RAM: 0 nothing, 1 half width, 2 full width (see `nest/gemma3n.py`).

| Function | Question it answers |
|---|---|
| need(level)            | which FFN state each layer needs to run this size |
| for_deadline(...)      | the largest size predicted to finish in time |
| for_memory(...)        | the largest size whose weights fit in a share of this Mac's RAM |
"""


def need(level, n_layers=35):
    return [0 if i in level["skip"] else level["blocks"][i] for i in range(n_layers)]


def union_need(levels, n_layers=35):
    return [max(col) for col in zip(*[need(l, n_layers) for l in levels])]


def ffn_bytes(prof, states):
    return sum(prof["layer_bytes"][i][s] for i, s in enumerate(states))


def model_bytes(prof, states):
    """Weights in RAM for these layer states: the part that never changes plus the FFN part."""
    return prof["base_bytes"] + ffn_bytes(prof, states)


def disk_bytes(prof, level, resident):
    """Bytes that must be read from disk to run `level` when `resident` is in RAM."""
    return sum(prof["layer_read"][i] for i, n in enumerate(need(level, len(resident))) if n > resident[i])


def predict_s(prof, level, prompt_tokens, max_tokens, disk=0):
    """Predicted seconds for one answer. The margin covers other apps using the GPU at the same time."""
    p = prof["levels"][level["id"]]
    return prof["margin"] * (prompt_tokens / p["prefill_tps"] + max_tokens / p["decode_tps"]) + disk / prof["disk_bps"]


def _largest_first(levels):
    return sorted(levels, key=lambda l: (-l["pct"], -l["mmlu_pt"]))


def for_deadline(levels, prof, resident, prompt_tokens, max_tokens, deadline_s, allow_disk=False, cap_pct=100):
    """The largest size predicted to finish within deadline_s.

    Sizes that need a disk read are skipped unless allow_disk (the disk time counts in the prediction).
    If no size is fast enough, the fastest runnable size is returned with met=False.
    Returns {"level", "predicted_s", "disk_bytes", "met"}.
    """
    pool = [l for l in levels if l["pct"] <= cap_pct + 0.5] or [min(levels, key=lambda l: l["pct"])]
    rows = []
    for l in _largest_first(pool):
        d = disk_bytes(prof, l, resident)
        if d and not allow_disk:
            continue
        rows.append({"level": l, "disk_bytes": d, "predicted_s": predict_s(prof, l, prompt_tokens, max_tokens, d)})
    if not rows:                                   # nothing runs from RAM: the disk read is the only way
        for l in _largest_first(pool):
            d = disk_bytes(prof, l, resident)
            rows.append({"level": l, "disk_bytes": d, "predicted_s": predict_s(prof, l, prompt_tokens, max_tokens, d)})
    for r in rows:
        if r["predicted_s"] <= deadline_s:
            return {**r, "met": True}
    best = min(rows, key=lambda r: r["predicted_s"])
    return {**best, "met": False}


def for_memory(levels, prof, total_ram, memory_pct):
    """The largest size whose weights, plus working room, fit in memory_pct of total_ram.

    If none fit, the smallest size is returned with met=False.
    Returns {"level", "bytes", "budget", "met"}.
    """
    budget = total_ram * memory_pct / 100
    rows = [{"level": l, "bytes": model_bytes(prof, need(l)) + prof["work_bytes"]} for l in _largest_first(levels)]
    for r in rows:
        if r["bytes"] <= budget:
            return {**r, "budget": budget, "met": True}
    return {**min(rows, key=lambda r: r["bytes"]), "budget": budget, "met": False}
