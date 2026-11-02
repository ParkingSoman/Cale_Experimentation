"""How well does a speed table built from FEW measured sizes predict the sizes in between? (Install-time budget: fewer sizes = a shorter hardware test.)
Measures EVERY size for real (a few runs each), then builds tables from subsets of the sizes the way cale/tuning.py does and compares the predicted
answer and prompt speed with the real one.

usage: python scripts/sizes_accuracy.py HOME_DIR [--repeats 3] [--out results/sizes-accuracy-NAME.json]
Uses CALE_MODEL / CALE_BACKEND / CALE_LLAMA_LIB like scripts/cale_dev_server.py. HOME_DIR must be a new folder (nothing real is touched).
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    home = Path(a.home)
    if home.exists() and any(home.iterdir()):
        sys.exit("HOME_DIR must be new and empty")
    home.mkdir(parents=True, exist_ok=True)
    os.environ.update(CALE_HOME=str(home), CALE_DEBUG="1", CALE_PORT="47899")
    os.environ.setdefault("CALE_MODEL", str(ROOT / "models" / "gemma-3n-e4b-4bit"))
    from cale import paths
    from cale.engine import Engine
    from cale.tuning import Tuner
    from cale import gemma3n_facts as facts

    eng = Engine(paths.MODEL)
    eng._tuning = True
    with eng.lock:
        eng._load()
    tuner = Tuner(eng, "quick", lambda t: None, lambda t: None)
    tuner.warm_up()
    levels = sorted(eng.levels, key=lambda l: -l["pct"])
    real = {}
    t0 = time.time()
    for l in levels:
        r = tuner.point(l, eng.pred.table_prompt, answer=20, repeats=a.repeats)
        if r:
            real[l["id"]] = {"pre": r[0], "dec": r[1]}
    took = time.time() - t0
    full = max(levels, key=lambda l: l["pct"])
    small = min(levels, key=lambda l: l["pct"])
    by_id = {l["id"]: l for l in levels}

    def cost(l):
        return facts.cost(l, full)

    def predict(table, level, key):
        """Between two measured sizes by cost (what cale/predict.py _table_value does)."""
        pts = sorted((cost(by_id[i]), v[key]) for i, v in table.items())
        c = cost(level)
        lo = [p for p in pts if p[0] <= c]
        hi = [p for p in pts if p[0] >= c]
        if lo and hi:
            (c0, v0), (c1, v1) = lo[-1], hi[0]
            return v0 if c1 == c0 else v0 + (v1 - v0) * (c - c0) / (c1 - c0)
        c0, v0 = (lo[-1] if lo else hi[0])
        return v0 * (c0 / c) if key == "pre" else v0

    def pick(n):
        ordered = sorted(levels, key=lambda l: l["pct"])
        ids = {ordered[round(i * (len(ordered) - 1) / (n - 1))]["id"] for i in range(n)}
        ids |= {full["id"], small["id"]}
        return ids

    result = {"seconds_to_measure_all": round(took, 1), "sizes": len(real), "real": {k: [round(v["pre"], 1), round(v["dec"], 2)] for k, v in real.items()}, "subsets": {}}
    for n in (2, 3, 4, 5):
        ids = pick(n)
        table = {i: real[i] for i in ids if i in real}
        errs_dec, errs_pre = [], []
        for l in levels:
            if l["id"] in table or l["id"] not in real:
                continue
            errs_dec.append(predict(table, l, "dec") / real[l["id"]]["dec"] - 1)
            errs_pre.append(predict(table, l, "pre") / real[l["id"]]["pre"] - 1)
        if errs_dec:
            result["subsets"][str(n)] = {"measured": len(table), "left_out": len(errs_dec),
                                         "decode_mean_abs_err": round(statistics.mean(abs(e) for e in errs_dec), 4), "decode_max_abs_err": round(max(abs(e) for e in errs_dec), 4),
                                         "prompt_mean_abs_err": round(statistics.mean(abs(e) for e in errs_pre), 4), "prompt_max_abs_err": round(max(abs(e) for e in errs_pre), 4)}
    print(json.dumps(result["subsets"], indent=1))
    print("measured all sizes in", result["seconds_to_measure_all"], "s;", "repeats", a.repeats)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=1))
    with eng.lock:
        eng._unload("done")


if __name__ == "__main__":
    main()
