"""Per-layer embeddings on disk: same answers? how much RAM? how much slower?

Run: .venv/bin/python scripts/g3n_ple.py   (writes results/g3n-ple.json)
"""
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
MODEL = ROOT / "models" / "gemma-3n-e4b-4bit"
from g3n_gate_c import PROMPTS, N  # noqa: E402
from nest.gemma3n import NestedGemma3n  # noqa: E402


def run(m, sizes):
    texts, speeds = [], {}
    for lid in sizes:
        t = time.perf_counter()
        tok = 0
        for p in PROMPTS:
            r = m.generate(p, N, width=lid)
            tok += r["stats"]["tokens"]
            if lid == "e4b":
                texts.append(r["text"])
        speeds[lid] = round(tok / (time.perf_counter() - t), 1)
    return texts, speeds


def main():
    out = {}
    for mode in ("ram", "disk", "ram", "disk"):
        gc.collect()
        mx.clear_cache()
        m = NestedGemma3n.load(MODEL, width="e4b", ple_on_disk=(mode == "disk"))
        gb = round(mx.get_active_memory() / 1e9, 3)
        texts, speeds = run(m, ["e4b", "e2b"])
        out.setdefault(mode, []).append({"active_gb": gb, "tok_per_s": speeds, "texts": texts})
        del m
        gc.collect()
        mx.clear_cache()
    out["identical_text"] = out["ram"][0]["texts"] == out["disk"][0]["texts"]
    for mode in ("ram", "disk"):
        for r in out[mode]:
            r.pop("texts")
    json.dump(out, open(ROOT / "results" / "g3n-ple.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


main()
