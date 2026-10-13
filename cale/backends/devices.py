"""Which compute device runs the model. One rule for every computer. No device name and no layer count is written here.

A GPU helps only when the whole model fits in its own memory. Then every layer runs there, every size is faster, and
nothing moves between devices. If only part of the model fits, answers can be slower than on the CPU at the smaller
sizes (measured 2026-10-13: 4.0 and 3.8 against 9.5 and 11.8 tokens/s). So partial use is not automatic: a developer
asks for it. A GPU that shares memory with the CPU (type IGPU) gains nothing for answers, which are limited by memory
speed, so it is not chosen on its own either.

CALE_GPU (a setting for the person or the app that starts Cale):
  auto     (default) a GPU with its own memory, only if the whole model and its working memory fit
  off      the CPU alone
  partial  a GPU with its own memory, as many layers as llama.cpp's fit finds. Prompts may be faster, answers may be slower.
  <text>   the device whose name contains the text (any type). All layers if they fit, else llama.cpp's fit.
"""
import os

WORKING_BYTES = 800 << 20          # compute buffers, KV cache and room for the OS driver


def choose(devs, model_bytes, mode=None):
    """-> (device dict or None, n_gpu_layers, reason). n_gpu_layers: 999 = all, -1 = let llama.cpp's fit decide."""
    mode = (mode or os.environ.get("CALE_GPU") or "auto").strip()
    low = mode.lower()
    if low == "off":
        return None, 0, "the CPU (CALE_GPU=off)"
    own = sorted([d for d in devs if d["type"] == "GPU"], key=lambda d: -d["free"])
    if low in ("auto", "full"):
        for d in own:
            if d["free"] >= model_bytes + WORKING_BYTES:
                return d, 999, f"{d['name']} holds the whole model"
        return None, 0, "the CPU (no GPU with its own memory holds the whole model)"
    if low == "partial":
        if own:
            return own[0], -1, f"{own[0]['name']}, part of the model (CALE_GPU=partial)"
        return None, 0, "the CPU (no GPU with its own memory)"
    named = [d for d in devs if low in (d["name"] + " " + d["description"]).lower()]
    if named:
        d = max(named, key=lambda d: d["free"])
        return d, (999 if d["free"] >= model_bytes + WORKING_BYTES else -1), f"{d['name']} (CALE_GPU={mode})"
    return None, 0, f"the CPU (no device matches CALE_GPU={mode})"
