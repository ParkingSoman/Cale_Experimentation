"""MMLU-Pro letter scoring for Gemma 3n: nested sizes of one E4B file, or any plain model folder.

usage: g3n_bench.py nested  MODEL_DIR OUT.json [--levels e4b,e2b|all] [--n 1000]
       g3n_bench.py plain   MODEL_DIR OUT.json [--n 1000]          (e.g. the official E2B)
Same prompt and scoring as scripts/bench.py: the answer is started with "The answer is" and
the highest-scoring letter token wins. Writes accuracy and per-item picks.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mlx_vlm.models.gemma3n.language import logit_softcap  # noqa: E402

# Gemma 3n puts all its probability on "**" after "The answer is", so a letter compared there is noise.
# Two ways that do work (they give the same ranking of sizes; "ask" is the default):
#   ask    - end the question with "Answer with only the letter" and read the first token
#   bold   - start the answer with "The answer is **" and read the next token
MODES = {"ask": ("\n\nAnswer with only the letter of the correct option.", ""),
         "bold": ("", "The answer is **")}


def rows(n):
    f = ROOT / "data" / "bench" / ("mmlupro1k.jsonl" if n > 200 else "mmlupro.jsonl")
    return [json.loads(l) for l in f.read_text().splitlines()][:n]


def score(lm, proc, data, mode="ask"):
    tok = proc.tokenizer
    lid = {c: tok.encode(c, add_special_tokens=False)[0] for c in "ABCDEFGHIJ"}
    ask, prefix = MODES[mode]
    hits, items = 0, []
    for r in data:
        text = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": r["user"] + ask}]}],
                                        tokenize=False, add_generation_prompt=True) + prefix
        ids = mx.array(tok.encode(text, add_special_tokens=False)[-4096:])[None]
        h = lm.model(ids, cache=lm.make_cache())          # layers 20+ reuse earlier KV, so a cache is required
        lg = logit_softcap(lm.final_logit_softcapping, lm.model.embed_tokens.as_linear(h[:, -1:]))[0, 0]
        sc = lg[mx.array([lid[c] for c in r["choices"]])]
        pick = r["choices"][int(mx.argmax(sc).item())]
        hits += pick == r["answer"]
        items.append({"id": r["id"], "pick": pick, "ok": pick == r["answer"]})
        mx.clear_cache()
    return {"acc": round(100 * hits / len(data), 1), "n": len(data), "items": items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["nested", "plain"])
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--levels", default="e4b,e2b")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--mode", choices=list(MODES), default="ask")
    a = ap.parse_args()
    data, res = rows(a.n), {}
    mx.set_cache_limit(1024 ** 3)
    if a.kind == "plain":
        from mlx_vlm import load
        model, proc = load(a.model)
        t = time.perf_counter()
        res["plain"] = score(model.language_model, proc, data, a.mode)
        res["plain"]["seconds"] = round(time.perf_counter() - t)
        print("plain", res["plain"]["acc"], flush=True)
    else:
        from nest.gemma3n import NestedGemma3n
        m = NestedGemma3n.load(a.model)
        ids = [l["id"] for l in m.levels] if a.levels == "all" else a.levels.split(",")
        for lid in ids:
            m.set(lid)
            t = time.perf_counter()
            res[lid] = score(m.lm, m.processor, data, a.mode)
            res[lid]["seconds"] = round(time.perf_counter() - t)
            print(lid, res[lid]["acc"], f'({res[lid]["seconds"]}s)', flush=True)
            Path(a.out).write_text(json.dumps(res))
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
