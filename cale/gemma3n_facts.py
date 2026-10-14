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


ATTENTION_SHARE = 0.25             # the work of a layer apart from its feed-forward part, in half-width feed-forward units (about 12M against 50M weights)
HOLD_LAST_DEFAULT = 16             # policy: with no measured speeds yet, read the last prompt token alone when the last piece has at least this many tokens


def shared_share(level):
    """Share of the work for one prompt token that sits in the layers without their own key/value memory (layers FIRST_SHARED and up).
    A step with no output token skips them (llama/patches/0003), so a prompt reads this much faster. Layers differ only by their feed-forward width."""
    def work(i):
        return 0.0 if i in level["skip"] else ATTENTION_SHARE + level["blocks"][i]
    total = sum(work(i) for i in range(N_LAYERS))
    return sum(work(i) for i in range(FIRST_SHARED, N_LAYERS)) / total if total else 0.0


def hold_last_min(level, prefill_tps, decode_tps):
    """The shortest last prompt piece (tokens) for which reading its last token alone is faster than reading it in the piece.
    Reading the last token alone costs one step (every weight read once: 1 / answer speed). It saves the share `f` of the work of the
    other tokens of the piece, which the speeds `prefill_tps` (measured with the shortcut on) and `decode_tps` tell:
        m / ((1 - f) * p) > (m - 1) / p + 1 / a     which gives     m > (1 - f) / f * (p / a - 1)
    Returns 0 (never) when there is nothing to gain."""
    f = shared_share(level)
    if f <= 0.0 or prefill_tps <= 0.0 or decode_tps <= 0.0:
        return 0
    import math
    return max(2, math.ceil((1.0 - f) / f * max(prefill_tps / decode_tps - 1.0, 0.0)))


SCRATCH_BYTES = 500_000_000        # working memory for reading a prompt (activations and the engine's own buffers)
HOLD_CONTEXT = 4096                # context assumed when a hold asks for a memory share and no call is known yet
LIGHT_MAX_PCT = 75                 # `power: "light"` on battery or when hot: no size above this share of the full size
