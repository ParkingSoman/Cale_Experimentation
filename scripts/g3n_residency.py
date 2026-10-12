"""Residency check: memory per held size, exactness at full size, cost of reading missing weights.

Run: .venv/bin/python scripts/g3n_residency.py   (writes results/g3n-residency.json)
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


def gb():
    return round(mx.get_active_memory() / 1e9, 3)


def main():
    m = NestedGemma3n.load(MODEL, width="e4b")
    out = {"load_gb": gb(), "ffn_gb": round(m.ffn_bytes() / 1e9, 3)}
    ref = [m.generate(p, N)["text"] for p in PROMPTS]
    ref_e2b = [m.generate(p, N, width="e2b")["text"] for p in PROMPTS]     # smallest size with everything resident
    ids = [l["id"] for l in m.levels]
    out["sizes"] = {}
    for lid in ids:                                   # memory when exactly this size is held
        freed = m.reside([lid])
        out["sizes"][lid] = {"active_gb": gb(), "ffn_gb": round(m.ffn_bytes() / 1e9, 3),
                             "predicted_ffn_gb": round(m.size_bytes(m.resolve(lid)) / 1e9, 3)}
        m.reside(["e4b"])                             # back to full only to compare the next one from the same start
        m.ensure(m.resolve("e4b"))
    # hold the smallest, then ask for the full size
    m.reside(["e2b"])
    held_e2b = [m.generate(p, N, width="e2b")["text"] for p in PROMPTS]     # smallest size running from the owned copies
    out["e2b_same_after_hold"] = sum(a == b for a, b in zip(ref_e2b, held_e2b))
    out["hold_e2b_gb"] = gb()
    t = time.perf_counter()
    txt, disk = [], []
    for p in PROMPTS:
        r = m.generate(p, N, width="e4b")
        txt.append(r["text"])
        disk.append((r["stats"]["disk_mb"], r["stats"]["disk_s"], r["stats"]["seconds"]))
    out["full_from_e2b"] = {"per_call_disk_mb_disk_s_total_s": disk, "seconds_all": round(time.perf_counter() - t, 2), "peak_gb": round(mx.get_peak_memory() / 1e9, 3),
                            "active_after_gb": gb(), "same_as_resident_full": sum(a == b for a, b in zip(ref, txt))}
    json.dump(out, open(ROOT / "results" / "g3n-residency.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


main()
