"""Drift check: compare a model level with saved predictions of the full model.

The full model runs once. For each held-out sample we keep its top-K next-token
log-probabilities at every text position. A level is then scored by:
  kl     - KL(full || level) over the full model's top-K tokens (lower is better)
  top1   - share of positions where both pick the same most likely token
"""
import os
from collections import defaultdict

import mlx.core as mx
import numpy as np

from .io import encode, hidden, logprobs_at

K = 64


def _text_positions(ids, kw):
    """Positions t whose NEXT token is a text token (not an image soft token)."""
    L = ids.shape[1]
    tt = kw.get("mm_token_type_ids")
    if tt is None:
        return np.arange(L - 1)
    nxt = np.array(tt[0, 1:])
    return np.nonzero(nxt == 0)[0]


def _forward_logprobs(model, ids, kw, pos):
    return logprobs_at(model, hidden(model, ids, kw), pos)


def save_reference(model, processor, recs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for r in recs:
        path = f"{out_dir}/{r['id']}.npz"
        if os.path.exists(path):
            continue
        ids, kw = encode(processor, r)
        pos = _text_positions(ids, kw)
        lp = _forward_logprobs(model, ids, kw, pos)
        top = mx.argpartition(-lp, K, axis=-1)[:, :K]
        vals = mx.take_along_axis(lp, top, axis=-1)
        mx.eval(top, vals)
        np.savez(path, pos=pos, idx=np.array(top).astype(np.int32), lp=np.array(vals).astype(np.float16))


def score(model, processor, recs, ref_dir, chunk=256):
    """Return {source: {"kl": mean KL, "top1": agreement}} plus an "all" row."""
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    for r in recs:
        ref = np.load(f"{ref_dir}/{r['id']}.npz")
        ids, kw = encode(processor, r)
        h = hidden(model, ids, kw)
        mx.eval(h)
        pos = ref["pos"]
        kl_sum, agree_sum = 0.0, 0.0
        for i in range(0, len(pos), chunk):
            hc = h[0][mx.array(pos[i:i + chunk])]
            lg = model.language_model.logits_from_hidden(hc).astype(mx.float32)
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            idx = mx.array(ref["idx"][i:i + chunk])
            p_lp = mx.array(ref["lp"][i:i + chunk].astype(np.float32))
            q_lp = mx.take_along_axis(lp, idx, axis=-1)
            kl = mx.sum(mx.exp(p_lp) * (p_lp - q_lp), axis=-1).sum()
            ref_top = mx.take_along_axis(idx, mx.argmax(p_lp, axis=-1, keepdims=True), axis=-1)[:, 0]
            agree = (mx.argmax(lp, axis=-1) == ref_top).sum()
            mx.eval(kl, agree)
            kl_sum += float(kl); agree_sum += float(agree)
        del h
        mx.clear_cache()
        for key in (r["source"], "all"):
            a = acc[key]
            a[0] += kl_sum; a[1] += agree_sum; a[2] += len(pos)
    return {k: {"kl": v[0] / v[2], "top1": v[1] / v[2], "n": v[2]} for k, v in acc.items()}


def fmt(res):
    keys = sorted(k for k in res if k != "all" and not k.startswith("_")) + ["all"]
    return " | ".join(f"{k} kl={res[k]['kl']:.3f} t1={res[k]['top1']:.3f}" for k in keys)
