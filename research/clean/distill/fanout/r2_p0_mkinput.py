"""P0 round 2: fast variants of the self-made generic input (ResNet-18 first). Own file; imports make_input read-only.

Object (picked by): the 256-image generic input X of the floor (picked by the board: X is the one input every
activity rule reads). Each variant changes ONE knob of make_input's recipe and names it.

Knobs:
  --K, --lr                 steps and Adam step size (make_input: 32, .05)
  --target ce   : CE to y ~ softmax(fc.bias), as make_input
           ent  : minimise the entropy of the network's own prediction (no drawn target; the network picks the class)
           act  : (r18 and clip) maximise the mean squared post-ReLU activation of every residual block output, each block's term
                  divided by its value at step 0 (a "wake every layer" target; no BatchNorm stat is read)
  --div w       : batch-statistics-free diversity: + w * mean pairwise cosine of the pooled 512-d features of the
                  batch (pushes images apart; reads no running_mean / running_var)
  CLIP act: the block term is the mean relu(z)^2 of the MLP pre-activation z = fc(ln2(s)), all tokens (the units
  whose firing rate on pink is .35 of the photo rate), each divided by its step-0 value.
  --color       : learn a 3x3 colour mixing per image (init identity), applied before chan_norm. The network chooses
                  how the channels co-vary. No photo statistic is used.
Null (--kind null): the same recipe on the shuffled-weights copy (make_input.shuffle_r18).

PHOTO USE: none. BatchNorm stats: never read (the model runs in eval mode, which APPLIES them as part of the map).
Output: cache/r18_<tag>_n256_s1.pt (+ .json).
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_input as mi  # noqa: E402

DEV = mi.DEV


def pooled_and_blocks(A, x):
    feats = []
    h = A.maxpool(A.relu(A.bn1(A.conv1(x))))
    for layer in (A.layer1, A.layer2, A.layer3, A.layer4):
        for blk in layer:
            h = blk(h)
            feats.append(h)
    p = torch.flatten(A.avgpool(h), 1)
    return A.fc(p), p, feats


def clip_acts(T, x):
    """CLIP: per block, the MLP pre-activation z = fc(ln2(s)) (all tokens). Same ops as vitgen.block."""
    import vitgen
    s = vitgen.embed(T, x)
    zs = []
    for B in T["blocks"]:
        s = s + vitgen.attn(vitgen.norm(s, B["ln1"]), B)
        z = F.linear(vitgen.norm(s, B["ln2"]), B["fc_w"], B["fc_b"])
        zs.append(z)
        s = s + F.linear(F.gelu(z), B["pr_w"], B["pr_b"])
    return zs


def run(a):
    if a.net == "clip":
        A, y = mi.load_net("clip", a.kind)
        y = torch.zeros(256, dtype=torch.long)
    else:
        A, y = mi.load_net("r18", a.kind)
    x_all = torch.load(mi.PINK, weights_only=True)
    amp = mi.amp()
    gen = torch.Generator().manual_seed(mi.SEED)
    outs, log = [], []
    t_start = time.time()
    for bi, s in enumerate(range(0, len(x_all), mi.BS)):
        x0 = x_all[s:s + mi.BS].float()
        z = mi.z_start(x0, amp).requires_grad_(True)
        params = [z]
        M = None
        if a.color:
            M = torch.eye(3).repeat(len(x0), 1, 1).requires_grad_(True)
            params.append(M)
        with torch.no_grad():
            g0 = mi.render(z, amp)
        opt = torch.optim.Adam(params, lr=a.lr)
        yb = y[s:s + mi.BS].to(DEV)
        base = None
        losses = []

        def image():
            x = x0 + (mi.render(z, amp) - g0)
            if M is not None:
                x = mi.chan_norm(torch.einsum("bij,bjhw->bihw", M, x))
            return x

        for _ in range(a.K):
            xt = mi.transform(image(), gen).to(DEV)
            if a.net == "clip":
                assert a.target == "act" and a.div == 0, "clip: only the act target here"
                e = torch.stack([F.relu(z.float()).pow(2).mean() for z in clip_acts(A, xt)])
                if base is None:
                    base = e.detach()
                L = -(e / base).mean()
                opt.zero_grad(set_to_none=True)
                L.backward()
                opt.step()
                losses.append(float(L.detach()))
                continue
            logits, p, feats = pooled_and_blocks(A, xt)
            logits = logits.float()
            if a.target == "ce":
                L = F.cross_entropy(logits, yb)
            elif a.target == "ent":
                lp = F.log_softmax(logits, 1)
                L = -(lp.exp() * lp).sum(1).mean()
            else:
                e = torch.stack([f.float().pow(2).mean() for f in feats])
                if base is None:
                    base = e.detach()
                L = -(e / base).mean()
            if a.div > 0:
                pn = F.normalize(p.float(), dim=1)
                c = pn @ pn.T
                n = len(pn)
                L = L + a.div * (c.sum() - n) / (n * (n - 1))
            opt.zero_grad(set_to_none=True)
            L.backward()
            opt.step()
            losses.append(float(L.detach()))
        with torch.no_grad():
            xk = image().cpu()
            log.append(dict(batch=bi, loss1=losses[0], lossK=losses[-1],
                            rel_change=float((xk - x0).norm() / x0.norm())))
        outs.append(xk)
    X = torch.cat(outs).half().contiguous()
    assert torch.isfinite(X).all()
    path = os.path.join(mi.CACHE, f"{a.net}_{a.tag}_n256_s1.pt")
    torch.save(X, path)
    meta = dict(vars(a), photo_use="none", minutes=round((time.time() - t_start) / 60, 2),
                loss1=sum(r["loss1"] for r in log) / len(log), lossK=sum(r["lossK"] for r in log) / len(log),
                rel_change=sum(r["rel_change"] for r in log) / len(log))
    json.dump(meta, open(path[:-3] + ".json", "w"), indent=1)
    print(json.dumps(meta))
    print("saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--net", default="r18", choices=("r18", "clip"))
    ap.add_argument("--kind", default="self", choices=("self", "null"))
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--target", default="ce", choices=("ce", "ent", "act"))
    ap.add_argument("--div", type=float, default=0.0)
    ap.add_argument("--color", action="store_true")
    run(ap.parse_args())
