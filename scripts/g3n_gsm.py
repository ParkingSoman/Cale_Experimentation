"""GSM8K (grade-school maths with reasoning) on chosen Gemma 3n sizes. Also records the real speed per size.

usage: g3n_gsm.py MODEL_DIR OUT.json [--levels e4b,e2b|all] [--n 100]
The model reasons step by step and ends with "The answer is N". Greedy decoding, up to 320 new tokens.
Unlike the letter test, this separates big sizes from small ones: the model has to think, not guess.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def last_number(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None


def correct(pred, answer):
    m = re.search(r"answer is\s*\$?(-?[\d,]*\.?\d+)", pred)
    got = m.group(1).replace(",", "") if m else last_number(pred)
    try:
        return got is not None and abs(float(got) - float(answer)) < 1e-6
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--levels", default="e4b,e2b")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (ROOT / "data" / "bench" / "gsm8k.jsonl").read_text().splitlines()][: a.n]
    from nest.gemma3n import NestedGemma3n
    m = NestedGemma3n.load(a.model)
    ids = [l["id"] for l in m.levels] if a.levels == "all" else a.levels.split(",")
    res = {}
    for lid in ids:
        m.set(lid)
        hits, toks, secs, items = 0, 0, 0.0, []
        for r in rows:
            out = m.generate(r["user"], max_tokens=320)
            ok = correct(out["text"], r["answer"])
            hits += ok
            toks += out["stats"]["tokens"]
            secs += out["stats"]["seconds"]
            items.append({"id": r["id"], "ok": bool(ok)})
        res[lid] = {"acc": round(100 * hits / len(rows), 1), "n": len(rows), "tok_per_s": round(toks / secs, 1),
                    "seconds": round(secs), "items": items}
        print(lid, res[lid]["acc"], f'{res[lid]["tok_per_s"]} tok/s', f'({res[lid]["seconds"]}s)', flush=True)
        Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
