"""Pair study, step 3 done properly: a parent's children found by removing the parent (Base only).

For 20 confident pairs (s, p) with p in stage-4 block 0, spread over match strength:
  p's top 64 fit photos (by per-photo peak, tools_peaks_3000.npz).
  Base forward twice: normal, and with p's hidden output (after GELU) replaced by its mean.
  Child score of a later neuron c (blocks 1, 2): mean drop of c's pre-activation where p is on, divided by c's std.
  Children = the 6 largest positive scores. Also the size of p's effect inside the stage against its direct effect
  on the class scores (is there a tree inside stage 4 at all?).
"""
import json
import os

import numpy as np
import torch

from cache import DEV, OUT, load, net, split

torch.set_grad_enabled(False)


class Fill:
    """Replace one hidden neuron's GELU output in a block with a constant."""

    def __init__(self, m, block):
        self.p, self.val = None, None
        m.features[7][block].block[4].register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        if self.p is None:
            return out
        out = out.clone()
        out[..., self.p] = self.val
        return out


def pick_pairs(k=20):
    tree = json.load(open(os.path.join(OUT, "tree.json")))
    pr = [q for q in tree["pairs"] if q["block"] == 0]
    pr = sorted(pr, key=lambda q: q["fire"] + q["write"])
    ix = np.linspace(len(pr) // 4, len(pr) - 1, k).astype(int)  # upper three quarters of match strength
    return [pr[i] for i in ix]


def main():
    pairs = pick_pairs()
    pk = np.load(os.path.join(OUT, "tools_peaks_3000.npz"))
    idx3000, pb = pk["idx"], pk["base"].astype(np.float32)
    ph = net.Photos("read")
    m = load("base")
    st = {}
    for b, blk in enumerate(m.features[7]):
        blk.block[3].register_forward_hook(lambda mod, i, o, b=b: st.__setitem__(b, o))
        blk.block[4].register_forward_hook(lambda mod, i, o, b=b: st.__setitem__(f"g{b}", o))
    fill = Fill(m, 0)
    # means of every block-0 GELU output, over 500 fit photos
    fit = split(ph.lab)
    mi = np.sort(np.random.default_rng(2).choice(np.where(fit)[0], 500, replace=False))
    acc = torch.zeros(4096, device=DEV)
    for k in range(0, 500, 50):
        m(ph.tensor(mi[k:k + 50]).to(DEV))
        acc += st["g0"].mean((0, 1, 2))
    gmean = (acc / 10).cpu()
    out, full = [], []
    for q in pairs:
        p = q["p"]
        top = idx3000[np.argsort(-pb[:, 0, p])[:64]]
        x = ph.tensor(np.sort(top)).to(DEV)
        fill.p = None
        y0 = m(x)
        a = {b: st[b].clone() for b in (0, 1, 2)}
        fill.p, fill.val = p, float(gmean[p])
        y1 = m(x)
        on = (a[0][..., p] > 0)  # positions where p is on
        res = dict(s=q["s"], p=p, fire=q["fire"], write=q["write"], kids=[])
        tot, full_sc = {}, {}
        for b in (1, 2):
            d = (a[b] - st[b])  # drop when p is removed
            sd = a[b].reshape(-1, 4096).std(0) + 1e-6
            sc = (d[on].mean(0) / sd).cpu().numpy()
            tot[b] = float(np.abs(sc).mean())
            full_sc[b] = sc
            for c in np.argsort(-sc)[:6]:
                res["kids"].append(dict(block=b, c=int(c), score=float(sc[c])))
        res["kids"] = sorted(res["kids"], key=lambda d: -d["score"])[:6]
        full.append(np.concatenate([full_sc[1], full_sc[2]]))
        res["mean_abs_score_b1"], res["mean_abs_score_b2"] = tot[1], tot[2]
        dl = (y0 - y1)
        res["logit_change_rms"] = float(dl.pow(2).mean().sqrt())
        res["logit_change_top_class"] = float(dl.gather(1, y0.argmax(1, keepdim=True)).mean())
        fill.p = None
        out.append(res)
        print(p, [(d["block"], d["c"], round(d["score"], 2)) for d in res["kids"]],
              f"mean|score| b1 {tot[1]:.3f} b2 {tot[2]:.3f}, top-class logit drop {res['logit_change_top_class']:.3f}",
              flush=True)
    json.dump(out, open(os.path.join(OUT, "kids.json"), "w"), indent=1)
    np.save(os.path.join(OUT, "kids_scores.npy"), np.stack(full).astype(np.float32))  # (pairs, 8192): blocks 1, 2


if __name__ == "__main__":
    main()
