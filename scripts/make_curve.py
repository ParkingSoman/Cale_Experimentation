"""Merge quality (any-L.json, nest-L.json) and speed/RAM (speed/curve.json) into results/curve.json.
Cale serves this file at /v1/models so an app can show what each size costs and gives.
"""
import json
from pathlib import Path

R = Path("results")
anyl = json.loads((R / "bench/any-L.json").read_text()) if (R / "bench/any-L.json").exists() else {}
nestl = json.loads((R / "bench/nest-L.json").read_text())
speed = json.loads((R / "speed/curve.json").read_text()) if (R / "speed/curve.json").exists() else {}


def acc(level, task):
    for src in (anyl, nestl):
        v = src.get(level, {}).get(task)
        if v:
            return round(v["acc"] * 100, 1)
    return None


rows = []
for pct in (100, 90, 80, 70, 60, 50):
    s = speed.get(str(pct), {})
    rows.append({"size": pct, "mmlu_pro": acc(f"w{pct}", "mmlupro1k"), "mmstar": acc(f"w{pct}", "mmstar300"),
                 "chat_tok_s": s.get("chat_tps"), "chat_tok_s_draft": s.get("chat_draft_tps"),
                 "math_tok_s": s.get("math_tps"), "math_tok_s_draft": s.get("math_draft_tps"),
                 "ram_gb": s.get("resident_gb")})
out = {"model": "gemma-12b", "quality_note": "MMLU-Pro n=1000, MMStar n=300, no healing", "rows": rows}
(R / "curve.json").write_text(json.dumps(out, indent=1))
print(json.dumps(rows, indent=1))
