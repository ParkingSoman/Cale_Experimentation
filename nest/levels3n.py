"""The official Gemma 3n sizes (Google's MatFormer table) and the rule that maps a percent to one.

Pure Python on purpose: the Cale server reads this without loading MLX.
A size is 'blocks' (1 = half-width FFN, 2 = full) for each of the 35 E4B layers, plus layers to skip.
"""
import json
from pathlib import Path

BLOCK = 8192                    # FFN channels per block; every official size uses 8192 or 16384 per layer
FULL_PARAMS_B = 3.98            # effective parameters of E4B; a size's percent is measured against it
LEVELS_FILE = Path(__file__).with_name("gemma3n_levels.json")
ALIASES = {"smart": "max", "fast": "min"}


def load_levels(n_layers=35):
    """All official sizes, largest first: dicts with id, name, pct, eff_params_b, mmlu_pt, blocks, skip."""
    out = []
    for row in json.loads(LEVELS_FILE.read_text()):
        skip = set(row["skip"])
        kept = [i for i in range(n_layers) if i not in skip]
        assert len(kept) == len(row["ffn"]), row["name"]
        blocks = [2] * n_layers
        for i, w in zip(kept, row["ffn"]):
            blocks[i] = w // BLOCK
        name = row["name"]
        if name == "Main model":
            lid = "e4b"
        elif name.startswith("official"):
            lid = "e2b"
        else:
            kind = "block" if "block-level" in name else "layer"
            lid = f"e{row['eff_params_b']:.2f}-{kind}"
        out.append({"id": lid, "name": name, "eff_params_b": row["eff_params_b"], "mmlu_pt": row["mmlu_pt"],
                    "pct": round(100 * row["eff_params_b"] / FULL_PARAMS_B), "blocks": blocks, "skip": skip})
    return sorted(out, key=lambda l: (-l["eff_params_b"], l["id"]))


def nearest(levels, target_pct):
    """The size whose percent is closest to target_pct. A tie goes to the better MMLU score."""
    return min(levels, key=lambda l: (abs(l["pct"] - target_pct), -l["mmlu_pt"]))


def pick(levels, target_pct, cap_pct=100):
    """The size that runs for a request: the nearest to target_pct among sizes at or under cap_pct.
    If the cap is below every size, the smallest size runs."""
    pool = [l for l in levels if l["pct"] <= cap_pct + 0.5] or [min(levels, key=lambda l: l["pct"])]
    return nearest(pool, min(target_pct, cap_pct))


def pct_range(levels):
    return min(l["pct"] for l in levels), max(l["pct"] for l in levels)


# Facts about Gemma 3n that the runtime may rely on (read from models/gemma-3n-e4b-4bit/config.json).
MAX_CONTEXT = 32768                  # max_position_embeddings: the longest prompt + answer the model was trained for
KV_BYTES_PER_TOKEN = 8192            # 4 full-attention layers own a KV cache (layers 4, 9, 14, 19): 4 x 2 x 2 heads x 256 x 2 bytes.
                                     # The 16 sliding-window layers add a fixed 17 MB. The other 15 layers share these caches.
