"""What this computer can do: speed of every size, bytes of every size, speed of the disk.

Measured once when a relative goal first needs it (about 20 s), saved as profile.json in Cale's home,
and nudged by real answers afterwards (the median of the last few). A new computer or a new model file
means a new profile: the file records the model size and the computer's RAM and is ignored if they differ.
"""
import json
import statistics
import time

from . import backends, oslayer, paths

PROFILE = paths.HOME / "profile.json"
MARGIN = 1.25            # safety factor on predicted time: phase 2 saw other apps slow the GPU by up to 1.6x
WORK_BYTES = 400_000_000 # room for the KV cache and activations on top of the weights
KEEP = 7                 # real answers kept per size
PROMPT = "Write a short note about why the sky is blue, then list three facts about the ocean. " * 14


def total_ram():
    return oslayer.total_ram()


def model_key(model):
    return {"total_ram": total_ram(), "model_bytes": sum(f.stat().st_size for f in model.model_files()),
            "ple_on_disk": model.ple_on_disk, "backend": backends.name()}


def _line(points):
    """Least-squares line y = a + b*x through (x, y) points."""
    n = len(points)
    mx_, my = sum(x for x, _ in points) / n, sum(y for _, y in points) / n
    var = sum((x - mx_) ** 2 for x, _ in points)
    b = sum((x - mx_) * (y - my) for x, y in points) / var if var else 0.0
    return my - b * mx_, b


def measure(model, tokens=48, fast=None):
    """Time the sizes on this computer. The model must have every FFN in RAM. Returns a profile dict.

    fast: time only the full, a middle and the smallest size, and fit the time per answer token against the
    size's effective parameters for the rest. On a slow computer this is minutes saved. Real answers refine it later."""
    assert model.all_full(), "measure() needs the full model in RAM"
    fast = (backends.name() != "mlx") if fast is None else fast
    full = model.resolve("e4b")
    prefill_tps = None
    levels, ordered = {}, sorted(model.levels, key=lambda l: -l["pct"])
    timed = ordered
    if fast and len(ordered) > 3:
        mid = min(ordered[1:-1], key=lambda l: abs(l["pct"] - 75))
        timed = [ordered[0], mid, ordered[-1]]
    for lv in timed:
        r = model.generate("Count from 1 to 60 in words.", tokens, width=lv)["stats"]
        decode_tps = r["tokens"] / max(r["seconds"], 1e-6)
        if prefill_tps is None:
            long = model.generate(PROMPT, 1, width=full)["stats"]
            prefill_tps = long["prompt_tokens"] / max(long["seconds"], 1e-6)
        levels[lv["id"]] = {"pct": lv["pct"], "decode_tps": round(decode_tps, 2), "samples": []}
    if len(timed) < len(ordered):
        a, b = _line([(l["eff_params_b"], 1.0 / levels[l["id"]]["decode_tps"]) for l in timed])
        for lv in ordered:
            if lv["id"] not in levels:
                levels[lv["id"]] = {"pct": lv["pct"], "decode_tps": round(1.0 / max(a + b * lv["eff_params_b"], 1e-6), 2),
                                    "samples": []}
    top = levels[full["id"]]["decode_tps"]
    for v in levels.values():                    # prefill scales with the size too (same weights, same maths)
        v["prefill_tps"] = round(prefill_tps * v["decode_tps"] / top, 1)
    ffn_full = model.ffn_bytes()
    model.reside(["e2b"])                        # read the FFN back once to time the disk
    read, secs = model.ensure(full)
    model.reside(["e4b"])
    model.clear_cache()
    return {"version": 2, "key": model_key(model), "margin": MARGIN, "work_bytes": WORK_BYTES,
            "levels": levels, "layer_bytes": model.layer_bytes, "layer_read": model.layer_read,
            "base_bytes": int(model.active_bytes() - ffn_full), "disk_bps": read / max(secs, 1e-6),
            "disk_samples": [], "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def load(model):
    """The saved profile if it matches this computer and this model file, else None."""
    try:
        prof = json.loads(PROFILE.read_text())
        return prof if prof.get("version") == 2 and prof.get("key") == model_key(model) else None
    except (OSError, ValueError):
        return None


def save(prof):
    tmp = PROFILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(prof, indent=1))
    tmp.replace(PROFILE)


def observe(prof, level_id, tokens, seconds, prompt_tokens, disk_bytes=0, disk_s=0.0):
    """Fold one real answer into the profile. Short answers are too noisy to use."""
    p = prof["levels"].get(level_id)
    if p is None:
        return
    if disk_bytes > 50_000_000 and disk_s > 0:
        prof["disk_samples"] = (prof["disk_samples"] + [disk_bytes / disk_s])[-KEEP:]
        prof["disk_bps"] = statistics.median(prof["disk_samples"])
    if tokens >= 16:
        decode_s = seconds - prompt_tokens / p["prefill_tps"]
        if decode_s > 0:
            p["samples"] = (p["samples"] + [tokens / decode_s])[-KEEP:]
            if len(p["samples"]) >= 3:
                p["decode_tps"] = round(statistics.median(p["samples"]), 2)
