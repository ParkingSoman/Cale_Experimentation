"""Everything Cale knows about Gemma 3n, in one place. Pure Python: no MLX, no llama.cpp.

For this prototype Gemma 3n is the final model, so its facts are written down on purpose (docs/PRINCIPLES.md, rule 5).
A later model gets its own file. The predictor (cale/predict.py) and the size choice (cale/choose.py) read only this.

What makes speed predictable: producing one answer token reads every active weight once, so its time is bytes read
divided by memory speed. Reading a prompt is limited by arithmetic, which also grows with the active weights. Both
scale with a size's *effective parameters*, a number Google publishes for every official size (nest/gemma3n_levels.json).
"""
from nest.levels3n import BLOCK, FULL_PARAMS_B, KV_BYTES_PER_TOKEN, MAX_CONTEXT, load_levels, nearest, pick  # noqa: F401

N_LAYERS = 35
FIRST_SHARED = 20                  # layers 20 and up reuse the KV cache of earlier layers, so only they may be skipped
KV_FIXED_BYTES = 17_000_000        # the 16 sliding-window layers hold a fixed window of keys and values
BITS_PER_WEIGHT = 4.5              # llama.cpp Q4_0 and MLX 4-bit (group 64) both store 4.5 bits per weight


def levels(n_layers=N_LAYERS):
    """The sizes Cale can run, largest first. Same rule on every engine."""
    return [l for l in load_levels(n_layers) if min(l["skip"], default=99) >= FIRST_SHARED]


def full_level(lvls=None):
    return max(lvls or levels(), key=lambda l: l["eff_params_b"])


def kv_bytes(context_tokens):
    """Bytes of key/value cache for this many tokens of prompt plus answer."""
    return KV_FIXED_BYTES + int(context_tokens) * KV_BYTES_PER_TOKEN


def bytes_per_token(level):
    """Weight bytes read to produce one answer token at this size."""
    return level["eff_params_b"] * 1e9 * BITS_PER_WEIGHT / 8


def cost(level, full=None):
    """How much work this size is compared with the full size (1.0 = the full size). Time per token and per prompt token scale with it."""
    return level["eff_params_b"] / (full or full_level())["eff_params_b"]


def quality(level):
    """Share of the full size's next-word choices this size repeats (1.0 = the full size). Measured once, on held-out text."""
    import json
    from pathlib import Path
    global _QUALITY
    try:
        _QUALITY
    except NameError:
        _QUALITY = json.loads((Path(__file__).with_name("gemma3n_quality.json")).read_text())["sizes"]
    return _QUALITY.get(level["id"], {}).get("top1", 1.0)


SCRATCH_BYTES = 500_000_000        # working memory for reading a prompt (activations and the engine's own buffers)
HOLD_CONTEXT = 4096                # context assumed when a hold asks for a memory share and no call is known yet
LIGHT_MAX_PCT = 75                 # `power: "light"` on battery or when hot: no size above this share of the full size
