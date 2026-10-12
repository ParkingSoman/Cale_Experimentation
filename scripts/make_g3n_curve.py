"""Join the measurements into results/g3n-curve.json, which Cale shows in GET /v1/models.

Reads results/g3n-kl.json (closeness to E4B), results/g3n-speed.json (tok/s) and results/bench/g3n-gsm8k.json
(maths score, only the sizes that were run). A size is "dominated" when another size is no bigger and no
farther from E4B and no slower: it is still selectable, but never the best pick.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from nest.levels3n import load_levels  # noqa: E402


def load(p):
    f = ROOT / p
    return json.loads(f.read_text()) if f.exists() else {}


def main():
    kl, speed, gsm = load("results/g3n-kl.json"), load("results/g3n-speed.json"), load("results/bench/g3n-gsm8k.json")
    rows = []
    for lv in load_levels():
        k, s = kl.get("levels", {}).get(lv["id"], {}), speed.get("sizes", {}).get(lv["id"], {})
        rows.append({"id": lv["id"], "size_pct": lv["pct"], "layers": 35 - len(lv["skip"]),
                     "half_width_layers": sum(1 for i, b in enumerate(lv["blocks"]) if b == 1 and i not in lv["skip"]),
                     "google_mmlu_pretrained": lv["mmlu_pt"], "kl_vs_full": k.get("kl"), "top1_vs_full": k.get("top1"),
                     "tok_per_s": s.get("tok_per_s"), "gsm8k": gsm.get(lv["id"], {}).get("acc")})
    full = next((r["tok_per_s"] for r in rows if r["id"] == "e4b"), None)
    def val(r, k, missing):
        return missing if r[k] is None else r[k]

    for r in rows:
        r["speedup"] = round(r["tok_per_s"] / full, 2) if full and r["tok_per_s"] else None
        r["dominated"] = any(o is not r and o["size_pct"] <= r["size_pct"] and val(o, "kl_vs_full", 9) <= val(r, "kl_vs_full", 9)
                             and val(o, "tok_per_s", 0) >= val(r, "tok_per_s", 0)
                             and (o["size_pct"], o["kl_vs_full"], o["tok_per_s"]) != (r["size_pct"], r["kl_vs_full"], r["tok_per_s"])
                             for o in rows)
    out = {"model": "gemma-3n-e4b (4-bit MLX)", "ram_gb": speed.get("active_gb"),
           "note": "Every size reads the same weights. RAM does not change with size; speed and closeness to the full model do.",
           "sizes": rows}
    (ROOT / "results" / "g3n-curve.json").write_text(json.dumps(out, indent=1))
    print(f'{"id":14}{"pct":>5}{"kl":>8}{"top1":>7}{"tok/s":>7}{"x":>6}{"gsm8k":>7}  note')
    for r in rows:
        print(f'{r["id"]:14}{r["size_pct"]:>5}{r["kl_vs_full"] or 0:>8.3f}{r["top1_vs_full"] or 0:>7.3f}{r["tok_per_s"] or 0:>7.1f}'
              f'{r["speedup"] or 0:>6.2f}{r["gsm8k"] or 0:>7}  {"dominated" if r["dominated"] else ""}')


if __name__ == "__main__":
    main()
