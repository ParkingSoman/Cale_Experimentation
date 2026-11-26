"""N8 step 2: the training gradient at step 0 (the build B), split into named terms, on the block-28 circuit.

Photos: E2's set (train cache, first 3 photos of each class: 3,000 fit photos).
For every photo:
  e_k   = block k's own error, alone, on Base's input: y_k^B(u_Base) - y_k^Base(u_Base)        (30 blocks)
  L     = sum_k J_k e_k   (first-order logit error, one forward-mode pass through Base with every e_k injected)
  real  = z_B - z_Base    (B's real logit error)
  delta = d KL / d x_29 on B's own path (x_29 = stream after block 28)                        the real back-signal
  t1    = J_28^T F J_28 e_28      t2 = J_28^T F (L - J_28 e_28)      t3 = delta - t1 - t2       (J_28 on Base's path)
  F v   = p_Base * v - p_Base (p_Base . v)
Gradients of block 28's kept writes and reads, per term (exact split, linear in delta, B's own a, pre, u):
  G_W2 = sum_pos (ls * delta) a^T          G_W1 = sum_pos (GELU'(pre) * (W2^T (ls * delta))) u^T
Loss is the SUM over photos of KL (train_ref uses the batch mean: only a scale).
Q1: cos(L, real) on class-centred logits.  Q2: term norms.  Q3-Q4: survivor s and owner d*.  Q5-Q6: against T - B.
Usage: python grad0.py [--n_per_class 3] [--smoke]
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gap"))
import common as C  # noqa: E402
from train_ref import load_mlp_state  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "step")
os.makedirs(OUT, exist_ok=True)
G, S_MAIN, D_STAR = 28, 4023, 1581


def mlp_parts(blk, x):
    """u, pre, a, y of one block from its input stream x (B, C, H, W)."""
    h = blk.block[2](blk.block[1](blk.block[0](x)))        # dwconv, permute, LayerNorm -> u (B, H, W, C)
    pre = blk.block[3](h)
    a = F.gelu(pre)
    y = blk.layer_scale * blk.block[6](blk.block[5](a))     # (B, C, H, W)
    return h, pre, a, y


def run(m, x3, eps=None, upto_start=0, store=None):
    """Tail of m from stage 3's input. eps[g] is added to the stream after block g."""
    BL = C.blocks(m)
    x = x3
    for g, blk in enumerate(BL):
        if g == 27:
            x = m.features[6](x)
        u, pre, a, y = mlp_parts(blk, x)
        if store is not None:
            store[g] = (x, u, pre, a, y)
        x = x + y
        if eps is not None:
            x = x + eps[g]
    return m.classifier(m.avgpool(x))


def head_from(m, x29):
    """Block 29 and the head, from the stream after block 28."""
    blk = C.blocks(m)[29]
    x = x29 + mlp_parts(blk, x29)[3]
    return m.classifier(m.avgpool(x))


def fisher(pB, v):
    return pB * v - pB * (pB * v).sum(1, keepdim=True)


def centred_cos(a, b):
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return F.cosine_similarity(a, b, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_class", type=int, default=3)
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--state", default="", help="a checkpoint .pt (key 'state') to take the gradient at, instead of B")
    ap.add_argument("--tag", default="grad0")
    a = ap.parse_args()
    torch.set_grad_enabled(True)
    tag = a.tag + ("_smoke" if a.smoke else "")
    logf = open(os.path.join(OUT, f"{tag}.log"), "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    base = C.load("base")
    for p in base.parameters():
        p.requires_grad_(False)
    cal, tr = C.Cache("cal"), C.Cache("train")
    lab = C.net.Photos("read").lab
    t0 = time.time()
    calib = C.calibrate(base, (x for x, _ in cal.batches(25)))
    TB = torch.load(os.path.join(C.OUT, "T_build.pt"), weights_only=False)
    TT = torch.load(os.path.join(C.OUT, "T.pt"), weights_only=False)
    mB = copy.deepcopy(base)
    Ss = C.build(mB, calib, TB["frac"], TB["how"])
    ndiff = sum(len(set(s1.tolist()) ^ set(s2.tolist())) for s1, s2 in zip(Ss, TB["S"]))
    Ss = [torch.as_tensor(s) for s in TB["S"]]   # the saved state overwrites every narrow layer: T_build.pt survivors are the truth
    load_mlp_state(mB, TB["state"])
    mT = copy.deepcopy(mB)
    if a.state:                                   # the current point: a checkpoint on T's path (mB keeps the name)
        load_mlp_state(mB, torch.load(a.state, weights_only=False)["state"])
    load_mlp_state(mT, TT["state"])
    for p in list(mB.parameters()) + list(mT.parameters()):
        p.requires_grad_(False)
    log(f"setup {time.time() - t0:.0f}s, device {C.DEV}, local build differs from saved survivors in {ndiff} indices (saved used)")

    S = Ss[G].numpy()
    k_s = int(np.where(S == S_MAIN)[0][0])
    cls = lab[tr.ix]
    rows = np.sort(np.concatenate([np.where(cls == c)[0][:a.n_per_class] for c in range(1000)]))
    if a.smoke:
        rows = rows[::50]
    blB, blBase = C.blocks(mB)[G], C.blocks(base)[G]
    ls = blB.layer_scale.view(-1)
    W2 = blB.block[5].weight                                   # (C, k)
    k = W2.shape[1]
    TERMS = ("real", "t1", "t2", "t3", "ti", "tc", "tp")
    acc = {t: dict(W2=torch.zeros_like(W2), W1=torch.zeros_like(blB.block[3].weight)) for t in TERMS + ("even", "odd")}
    q1_cos, q1_norm, kl_lin, kl_real = [], [], [], []
    s_read_photo = {t: [] for t in TERMS}   # per photo: s's read gradient contribution (vector)
    dstar_peak, score_rows = [], []
    jvp_mode = None
    t0 = time.time()
    for i0 in range(0, len(rows), a.bs):
        x3, zB = tr.get(rows[i0:i0 + a.bs])
        pB = F.softmax(zB, 1)
        # Base path: own errors e_k, and Base's pre at block 28 (all 4,096 neurons, for d*)
        stB = {}
        with torch.no_grad():
            run(base, x3, store=stB)
            e = []
            for g, (blkB, blkb) in enumerate(zip(C.blocks(mB), C.blocks(base))):
                x, u, pre, aa, y = stB[g]
                yb = blkB.layer_scale * blkB.block[6](blkB.block[5](F.gelu(blkB.block[3](u))))
                e.append(yb - y)
            dstar_peak.append(stB[G][2][..., D_STAR].amax((1, 2)).cpu())
            x29 = stB[G][0] + stB[G][4]
        # L = sum_k J_k e_k and J_28 e_28, forward mode on Base's path (central differences if jvp is missing)
        zeros = [torch.zeros_like(t) for t in e]

        def f_all(*eps):
            return run(base, x3, eps=list(eps))

        def f_28(v):
            return head_from(base, x29 + v)
        with torch.no_grad():
            if jvp_mode is None:
                try:
                    torch.func.jvp(f_28, (zeros[G],), (e[G],))
                    jvp_mode = "jvp"
                except Exception as ex:  # noqa: BLE001
                    jvp_mode = "fd"
                    log(f"jvp unavailable ({type(ex).__name__}), central differences")
            if jvp_mode == "jvp":
                _, Lz = torch.func.jvp(f_all, tuple(zeros), tuple(e))
                _, J28 = torch.func.jvp(f_28, (zeros[G],), (e[G],))
            else:
                h = 1e-2
                Lz = (f_all(*[h * t for t in e]) - f_all(*[-h * t for t in e])) / (2 * h)
                J28 = (f_28(h * e[G]) - f_28(-h * e[G])) / (2 * h)
        # B's own path: real logits and the real back-signal at x_29
        stS = {}
        with torch.no_grad():
            run(mB, x3, store=stS)
        xs29 = (stS[G][0] + stS[G][4]).detach().requires_grad_(True)
        zS = head_from(mB, xs29)
        kl = (pB * (F.log_softmax(zB, 1) - F.log_softmax(zS, 1))).sum()
        delta, = torch.autograd.grad(kl, xs29)
        real = (zS - zB).detach()
        q1_cos.append(centred_cos(Lz, real).cpu())
        rc = real - real.mean(1, keepdim=True)
        Lc = Lz - Lz.mean(1, keepdim=True)
        q1_norm.append((Lc.norm(dim=1) / rc.norm(dim=1)).cpu())
        lp = F.log_softmax(zB, 1)
        kl_lin.append((pB * (lp - F.log_softmax(zB + Lz, 1))).sum(1).cpu())
        kl_real.append((pB * (lp - F.log_softmax(zS.detach(), 1))).sum(1).cpu())
        # terms 1 and 2: VJPs on Base's path from the logits back to x_29
        x29v = x29.detach().requires_grad_(True)
        zb2 = head_from(base, x29v)
        v1 = fisher(pB, J28)
        v2 = fisher(pB, Lz - J28)
        dp_ = F.softmax(zS.detach(), 1) - pB
        vi = fisher(pB, real - Lz)                     # interaction: the logit error that is not the linear sum
        vc = dp_ - fisher(pB, real)                    # softmax curvature: p_s - p_B is not F (z_s - z_B)
        t1, = torch.autograd.grad(zb2, x29v, v1, retain_graph=True)
        t2, = torch.autograd.grad(zb2, x29v, v2, retain_graph=True)
        ti, = torch.autograd.grad(zb2, x29v, vi, retain_graph=True)
        tc, = torch.autograd.grad(zb2, x29v, vc)
        terms = dict(real=delta, t1=t1, t2=t2, t3=delta - t1 - t2, ti=ti, tc=tc, tp=delta - t1 - t2 - ti - tc)
        # gradients of block 28's kept writes and reads, per term, on B's own a, pre, u
        _, u, pre, aa, _ = stS[G]
        gp = torch.autograd.functional.vjp(F.gelu, pre, torch.ones_like(pre))[1]   # GELU'(pre)
        for t, d in terms.items():
            dl = (ls[:, None, None] * d).permute(0, 2, 3, 1)               # (B, H, W, C)
            acc[t]["W2"] += torch.einsum("bhwc,bhwk->ck", dl, aa)
            back = gp * (dl @ W2)                                           # (B, H, W, k)
            acc[t]["W1"] += torch.einsum("bhwk,bhwi->ki", back, u)
            s_read_photo[t].append(torch.einsum("bhw,bhwi->bi", back[..., k_s], u).cpu())
            if t == "real":
                score_rows.append(back.abs().sum((1, 2)).cpu())
                half = "even" if (i0 // a.bs) % 2 == 0 else "odd"
                acc[half]["W2"] += torch.einsum("bhwc,bhwk->ck", dl, aa)
                acc[half]["W1"] += torch.einsum("bhwk,bhwi->ki", back, u)
        if i0 // a.bs % 20 == 0:
            log(f"  {i0 + len(x3)}/{len(rows)} photos, {time.time() - t0:.0f}s, jvp mode {jvp_mode}, "
                f"Q1 cos median so far {float(torch.cat(q1_cos).median()):.3f}")

    q1c, q1n = torch.cat(q1_cos), torch.cat(q1_norm)
    klL, klR = torch.cat(kl_lin), torch.cat(kl_real)
    res = dict(n_photos=len(rows), jvp_mode=jvp_mode,
               Q1=dict(cos_median=float(q1c.median()), cos_q25=float(q1c.quantile(.25)), cos_q75=float(q1c.quantile(.75)),
                       norm_ratio_median=float(q1n.median()), kl_linear=float(klL.mean()), kl_real=float(klR.mean())))
    log(f"Q1: cos(sum_k J_k e_k, real logit error) median {res['Q1']['cos_median']:.3f} "
        f"(IQR {res['Q1']['cos_q25']:.3f}-{res['Q1']['cos_q75']:.3f}), norm ratio {res['Q1']['norm_ratio_median']:.3f}, "
        f"KL of Base+linear {res['Q1']['kl_linear']:.3f} vs real {res['Q1']['kl_real']:.3f}")
    # Q2: term norms on the full-batch gradients of block 28
    q2 = {}
    for wn in ("W2", "W1"):
        R = acc["real"][wn]
        q2[wn] = {t: dict(norm=float(acc[t][wn].norm()), cos_with_real=float(F.cosine_similarity(
            acc[t][wn].flatten(), R.flatten(), dim=0)),
            share=float((acc[t][wn].flatten() @ R.flatten()) / R.norm() ** 2)) for t in TERMS}
        q2[wn]["halves_cos"] = float(F.cosine_similarity(acc["even"][wn].flatten(), acc["odd"][wn].flatten(), dim=0))
        log(f"Q2 {wn}: " + ", ".join(f"{t} |{q2[wn][t]['norm']:.3g}| cos {q2[wn][t]['cos_with_real']:+.3f} share {q2[wn][t]['share']:+.3f}"
                                      for t in TERMS) + f"; gradient on even vs odd batches cos {q2[wn]['halves_cos']:+.3f}")
    res["Q2"] = q2
    # Q3: s's read gradient on d*'s top-200 photos
    dp = torch.cat(dstar_peak)
    top = torch.argsort(-dp)[:min(200, len(dp) // 15)]
    srp = {t: torch.cat(v) for t, v in s_read_photo.items()}
    gtot = srp["real"].sum(0)
    proj = srp["real"] @ gtot / gtot.norm()                                   # each photo's push along the total
    share_top = float(proj[top].sum() / proj.sum())
    share_abs = float(srp["real"].norm(dim=1)[top].sum() / srp["real"].norm(dim=1).sum())
    res["Q3"] = dict(n_top=len(top), photo_share=len(top) / len(dp), share_of_push=share_top, share_of_abs=share_abs,
                     ratio_abs=share_abs / (len(top) / len(dp)),
                     term_share_on_top={t: float((srp[t][top].sum(0) @ srp['real'][top].sum(0)) /
                                                 srp['real'][top].sum(0).norm() ** 2) for t in TERMS[1:]})
    log(f"Q3: d* top {len(top)} photos ({len(top) / len(dp):.3f} of photos) carry {share_abs:.3f} of |s read grad| "
        f"(ratio {res['Q3']['ratio_abs']:.1f}), {share_top:.3f} of the push along the total; "
        f"terms on those photos {res['Q3']['term_share_on_top']}")
    # Q4: per survivor, the step-0 read-push score on d*'s photos, and the same score's ratio to all photos
    sc = torch.cat(score_rows)                                                # (photos, k)
    on_d = sc[top].sum(0)
    ratio = on_d / sc.sum(0).clamp_min(1e-30)
    e2 = json.load(open(os.path.join(C.OUT, "e2_reads_T.json")))["rows"]
    owners = [r["s"] for r in e2 if r["block"] == G and r["dstar"] == D_STAR]
    others = [r["s"] for r in e2 if r["block"] == G and r["dstar"] != D_STAR]

    def pct(v, ids):
        rank = torch.empty(k)
        rank[torch.argsort(v)] = torch.arange(k, dtype=torch.float) / (k - 1)
        return [float(rank[int(np.where(S == i)[0][0])]) for i in ids]
    res["Q4"] = dict(owners=owners, pct_score=pct(on_d, owners), pct_ratio=pct(ratio, owners),
                     others=others, others_pct_score=pct(on_d, others), others_pct_ratio=pct(ratio, others))
    log(f"Q4: owner-{D_STAR} survivors, percentile of score on d* photos: median "
        f"{np.median(res['Q4']['pct_score']):.3f}; of ratio (d* photos / all) {np.median(res['Q4']['pct_ratio']):.3f}. "
        f"Other E2 survivors of block {G}: {np.median(res['Q4']['others_pct_score']):.3f} / "
        f"{np.median(res['Q4']['others_pct_ratio']):.3f}")
    # Q5-Q6: step-0 gradient vs the total change T - B, for s and for the whole block
    bT = C.blocks(mT)[G]
    dW1 = (bT.block[3].weight - blB.block[3].weight)
    dW2 = (bT.block[5].weight - W2)
    q5 = {}
    for wn, dW, g0, W0 in (("W1", dW1, acc["real"]["W1"], blB.block[3].weight),
                           ("W2", dW2, acc["real"]["W2"], W2)):
        if wn == "W1":
            ds, gs, ws = dW[k_s], g0[k_s], W0[k_s]
        else:
            ds, gs, ws = dW[:, k_s], g0[:, k_s], W0[:, k_s]
        q5[wn] = dict(s_cos_minus_grad=float(F.cosine_similarity(ds, -gs, dim=0)),
                      s_cos_minus_sign=float(F.cosine_similarity(ds, -torch.sign(gs), dim=0)),
                      s_rel_change=float(ds.norm() / ws.norm()),
                      block_cos_minus_grad=float(F.cosine_similarity(dW.flatten(), -g0.flatten(), dim=0)),
                      block_cos_minus_sign=float(F.cosine_similarity(dW.flatten(), -torch.sign(g0).flatten(), dim=0)),
                      block_rel_change=float(dW.norm() / W0.norm()),
                      **{f"s_cos_minus_{t}": float(F.cosine_similarity(ds, -(acc[t][wn][k_s] if wn == "W1" else acc[t][wn][:, k_s]), dim=0))
                         for t in TERMS[1:]})
        log(f"Q5/Q6 {wn}: " + ", ".join(f"{kk} {v:+.3f}" for kk, v in q5[wn].items()))
    res["Q5_Q6"] = q5
    json.dump(res, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)
    torch.save(dict(acc={t: {w: v.cpu() for w, v in d.items()} for t, d in acc.items()}, dstar_peak=dp, score=sc,
                    s_read_photo=srp, rows=rows), os.path.join(OUT, f"{tag}.pt"))


if __name__ == "__main__":
    main()
