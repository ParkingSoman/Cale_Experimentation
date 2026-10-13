"""Which compute device runs the model. One rule for every computer. No device name and no layer count is written here.

Two ways a GPU helps.
1. The whole model fits in its own memory. Every layer runs there, every size is faster, and nothing moves between devices.
2. The model does not fit, but the library can read repacked weights back (llama/patches/0002). The weights stay in RAM,
   repacked for the CPU, and the CPU answers at its own speed at every size. The GPU takes only the work on long prompts
   and gets the weights it needs for each pass. Measured 2026-10-13 on a 2 GB card: prompts 33 against 16 tokens/s on the CPU,
   answers equal (7.0 against 6.8). Nothing gets slower. Only the full size is helped: smaller sizes are views of the weights and stay on the CPU.
Part of the model on the GPU (llama.cpp's fit) is not automatic: it made answers at smaller sizes slower than on the CPU
(measured 2026-10-13: 4.0 and 3.8 against 9.5 and 11.8 tokens/s). A developer asks for it with `partial`.
A GPU that shares memory with the CPU (type IGPU) gains nothing for answers, which are limited by memory speed, so it is
not chosen on its own either.

CALE_GPU (a setting for the person or the app that starts Cale):
  auto     (default) a GPU with its own memory: all layers if the whole model and its working memory fit, else no layers
           on it and the GPU reads prompts (way 2 above) when the library can do that
  off      the CPU alone
  partial  a GPU with its own memory, as many layers as llama.cpp's fit finds. Prompts may be faster, answers may be slower.
  <text>   the device whose name contains the text (any type). All layers if they fit, else llama.cpp's fit.
"""
import os

WORKING_BYTES = 800 << 20          # only if llama.cpp cannot project the fit (an old library): compute buffers, KV cache and room for the OS driver


def one_per_card(gpus):
    """A card that two backends see (CUDA and Vulkan on an NVIDIA card) is listed once, with the vendor's own backend: CUDA, Metal or HIP,
    not the generic Vulkan. Measured on a GeForce MX150 (2026-10-13): prompts 68 tokens/s with CUDA and 26 with Vulkan, answers equal."""
    best = {}
    for d in gpus:
        k = d.get("description") or d["name"]
        generic = d["name"].lower().startswith("vulkan")
        if k not in best or (generic < best[k][1]):
            best[k] = (d, generic)
    return [d for d, _ in best.values()]


def choose(devs, model_bytes, mode=None, readback=False, fits=None):
    """-> (device dict or None, n_gpu_layers, reason). n_gpu_layers: 999 = all, 0 = none (the GPU only reads prompts), -1 = let llama.cpp's fit decide."""
    mode = (mode or os.environ.get("CALE_GPU") or "auto").strip()
    low = mode.lower()
    if low == "off":
        return None, 0, "the CPU (CALE_GPU=off)"
    own = sorted(one_per_card([d for d in devs if d["type"] == "GPU"]), key=lambda d: -d["free"])
    def holds_all(d):
        """Does the whole model fit in this GPU? llama.cpp's own projection says (`fits`). If it cannot, a rough rule with a margin."""
        said = fits(d) if fits is not None else None
        return said if said is not None else d["free"] >= model_bytes + WORKING_BYTES

    if low in ("auto", "full"):
        for d in own:
            if holds_all(d):
                return d, 999, f"{d['name']} holds the whole model"
        if readback and own:
            return own[0], 0, f"{own[0]['name']} reads prompts, the weights stay in RAM for the CPU (the model does not fit in its memory)"
        return None, 0, "the CPU (no GPU with its own memory holds the whole model)"
    if low == "partial":
        if own:
            return own[0], -1, f"{own[0]['name']}, part of the model (CALE_GPU=partial)"
        return None, 0, "the CPU (no GPU with its own memory)"
    named = [d for d in devs if low in (d["name"] + " " + d["description"]).lower()]
    if named:
        d = max(named, key=lambda d: d["free"])
        return d, (999 if holds_all(d) else -1), f"{d['name']} (CALE_GPU={mode})"
    return None, 0, f"the CPU (no device matches CALE_GPU={mode})"
