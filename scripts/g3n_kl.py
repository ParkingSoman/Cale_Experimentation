"""How close is each Gemma 3n size to full E4B on real text? (and how well does each predict the reference?)

usage: g3n_kl.py MODEL_DIR OUT.json [--official E2B_DIR] [--n 160]
For each held-out chat/web/code/math/multilingual sample the prompt and reference answer are fed through
every size (teacher forcing). Per size we report, over the answer tokens:
  kl    mean KL(E4B || size) in nats   (0 = same predictions as E4B)
  top1  how often the size's top token equals E4B's top token
  nll   mean negative log-likelihood of the reference answer (lower = better)
With --official the official E2B checkpoint is scored the same way (its nll and kl vs E4B).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mlx_vlm.models.gemma3n.language import logit_softcap  # noqa: E402


def samples(n):
    rows = [json.loads(l) for l in (ROOT / "data" / "heldout.jsonl").read_text().splitlines()]
    return [r for r in rows if not r.get("image") and r.get("assistant")][:n]


def encode(proc, r):
    tok = proc.tokenizer
    text = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": r["user"]}]}],
                                    tokenize=False, add_generation_prompt=True)
    p = tok.encode(text, add_special_tokens=False)[-700:]
    a = tok.encode(r["assistant"], add_special_tokens=False)[:128]
    return mx.array(p + a)[None], len(p)


def answer_logprobs(lm, ids, start):
    """log-softmax over the vocabulary at the positions that predict the answer tokens."""
    h = lm.model(ids, cache=lm.make_cache())
    lg = logit_softcap(lm.final_logit_softcapping, lm.model.embed_tokens.as_linear(h[:, start - 1:-1]))[0]
    lg = lg.astype(mx.float32)
    return lg - mx.logsumexp(lg, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--official")
    ap.add_argument("--n", type=int, default=160)
    a = ap.parse_args()
    mx.set_cache_limit(1024 ** 3)
    from nest.gemma3n import NestedGemma3n
    m = NestedGemma3n.load(a.model)
    off = None
    if a.official:
        from mlx_vlm import load
        off = load(a.official)
    data = samples(a.n)
    acc = defaultdict(lambda: defaultdict(float))              # name -> metric sums
    by_src = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    t0 = time.perf_counter()
    for k, r in enumerate(data):
        ids, start = encode(m.processor, r)
        tgt = ids[0, start:]
        m.set("e4b")
        ref = answer_logprobs(m.lm, ids, start)
        ref_top = mx.argmax(ref, axis=-1)
        p_ref = mx.exp(ref)
        runs = [(lv["id"], lambda lv=lv: (m.set(lv["id"]), answer_logprobs(m.lm, ids, start))[1]) for lv in m.levels]
        if off:
            runs.append(("official-e2b", lambda: answer_logprobs(off[0].language_model, ids, start)))
        n_tok = int(tgt.shape[0])
        for name, fn in runs:
            lp = fn()
            kl = mx.sum(p_ref * (ref - lp), axis=-1)
            same = (mx.argmax(lp, axis=-1) == ref_top)
            nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
            s = mx.stack([mx.sum(kl), mx.sum(same.astype(mx.float32)), mx.sum(nll)]).tolist()
            for d in (acc[name], by_src[r["source"]][name]):
                d["kl"] += s[0]; d["top1"] += s[1]; d["nll"] += s[2]; d["n"] += n_tok
        mx.clear_cache()
        if k % 20 == 0:
            print(f"{k + 1}/{len(data)}  {time.perf_counter() - t0:.0f}s", flush=True)
    fin = lambda d: {"kl": round(d["kl"] / d["n"], 4), "top1": round(d["top1"] / d["n"], 4), "nll": round(d["nll"] / d["n"], 4)}
    out = {"n_samples": len(data), "tokens": int(acc["e4b"]["n"]), "levels": {n: fin(d) for n, d in acc.items()},
           "by_source": {s: {n: fin(d) for n, d in v.items()} for s, v in by_src.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f'{"size":16} {"kl":>7} {"top1":>6} {"nll":>6}')
    for n, v in out["levels"].items():
        print(f'{n:16} {v["kl"]:7.4f} {v["top1"]:6.3f} {v["nll"]:6.3f}')


if __name__ == "__main__":
    main()
