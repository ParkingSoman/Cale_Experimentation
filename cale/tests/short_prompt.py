"""A graphics card that reads prompts has a cost per pass that depends on the size. Short prompts then do not scale like long ones:
the predictor must use a curve at the smallest size too (cale/predict.py `_shape`, cale/tuning.py `prefill_curve`).
Run: python cale/tests/short_prompt.py   (from the repo folder; no model, no server). Every number here is made up.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cale import gemma3n_facts as facts  # noqa: E402
from cale.predict import Predictor  # noqa: E402

LEVELS = facts.levels()
FULL, SMALL = facts.full_level(LEVELS), min(LEVELS, key=lambda l: l['pct'])
FAIL = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def make(with_small):
    p = Predictor("llama", 28e9, levels=LEVELS)
    # at 256 tokens the small size reads SLOWER than the full size (as on a card whose copy cost does not shrink with the size)
    p.table = {FULL["id"]: {"pre": 130.0, "dec": 5.7}, SMALL["id"]: {"pre": 80.0, "dec": 10.0}}
    p.table_prompt, p.corr_pre, p.corr_dec = 256, 1.0, 1.0
    p.n_dec = p.n_pre = 2
    p.pre_curve = [(32, 23.0), (256, 130.0), (512, 160.0)]                      # full size: a short prompt is slow (fixed cost per pass)
    if with_small:
        p.pre_curve_small, p.small_id = [(32, 90.0), (256, 80.0), (512, 82.0)], SMALL["id"]   # small size: a short prompt is fast
    return p


old, new = make(False), make(True)
check("without the second curve the small size looks slower for a 50-token prompt (the old mistake)",
      old.first_token_s(SMALL, 50) > old.first_token_s(FULL, 50), f"{old.first_token_s(SMALL, 50):.2f} vs {old.first_token_s(FULL, 50):.2f}")
check("with the second curve the small size is faster for a 50-token prompt (as measured)",
      new.first_token_s(SMALL, 50) < new.first_token_s(FULL, 50), f"{new.first_token_s(SMALL, 50):.2f} vs {new.first_token_s(FULL, 50):.2f}")
check("at the table's prompt length both predictions still match the table",
      abs(new.prefill_tps(FULL, 256) - 130.0) < 1e-6 and abs(new.prefill_tps(SMALL, 256) - 80.0) < 1e-6)
mid = next(l for l in LEVELS if abs(facts.cost(l, FULL) - (1 + facts.cost(SMALL, FULL)) / 2) < 0.1)
check("a size between the two sits between them for a short prompt",
      new.prefill_tps(SMALL, 50) > new.prefill_tps(mid, 50) > new.prefill_tps(FULL, 50) or new.prefill_tps(FULL, 50) < new.prefill_tps(mid, 50) < new.prefill_tps(SMALL, 50))
print("ALL PASS" if not FAIL else "FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
