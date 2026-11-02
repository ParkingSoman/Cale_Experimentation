"""S6 blind judge: candidate-only measurements for one held-out target.
Usage: python judge.py 2_194
(1) position-aligned containment on 3000 images (3/class analysis set): target map vs each candidate map pooled
    (max and min) over the block of candidate cells under each target cell.
    on = z > 0.5 * (99.9th pct of per-image max); neg = z < 0.5 * (0.1th pct of per-image min).
    fwd = P(cand on | targ on), offside = P(cand neg | targ on), rev = P(targ on | cand on), breadth = P(cand on),
    lift = fwd / breadth, negfwd = P(cand on | targ neg).
(2) ablation: zero the candidate's hidden unit everywhere, recompute target z at its top-20 spots (median drop
    fraction, all paths) and at its 20 most negative spots."""
import sys, os, json
import numpy as np, mlx.core as mx, mlx.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "vision2"))
from convnext import norm_img
import gallery as G

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
key = sys.argv[1]; C = json.load(open(os.path.join(D, "s6", "candidates.json")))[key]
tb, tn = map(int, key.split("_")); T = (tb, tn); cands = {c: tuple(v) for c, v in C.items()}
G._init(); m = G._m; X, IDX = G._X, G._idx; ST = m.stage_of; RES = lambda b: 56 // 2 ** ST[b]


def maps(imgs, targets, kill=None):
    """{(b,n): (N,R,R)} fc1 pre-activation maps; kill=(b,n) zeroes that hidden unit (post-GELU) in its block."""
    want = {}
    for b, n in targets: want.setdefault(b, []).append(n)
    out = {t: [] for t in targets}
    for i in range(0, len(imgs), 50):
        x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(imgs[i:i + 50])))))
        for b in range(max(want) + 1):
            if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
            blk = m.blocks[b]; z = blk.fc1(blk.pre(x))
            if b in want:
                zz = np.array(z[..., want[b]])
                for j, n in enumerate(want[b]): out[(b, n)].append(zz[..., j])
            if b == max(want): break
            h = nn.gelu(z)
            if kill is not None and kill[0] == b:
                msk = np.ones(h.shape[-1], np.float32); msk[kill[1]] = 0; h = h * mx.array(msk)
            x = x + blk.gamma * blk.fc2(h)
    return {t: np.concatenate(v) for t, v in out.items()}


N = len(IDX); allt = [T] + list(cands.values())
Zt = None; acc = {c: {"tm": [], "cmax": [], "cmin": []} for c in cands}; tm_all = []
for s in range(0, N, 500):
    imgs = np.asarray(X[IDX[s:s + 500]]); Z = maps(imgs, allt); zt = Z[T]; tm_all.append(zt)
    Rt = RES(tb)
    for c, p in cands.items():
        zp = Z[p]; f = RES(p[0]) // Rt
        if f >= 1:
            B = zp.shape[0]; zr = zp.reshape(B, Rt, f, Rt, f); acc[c]["cmax"].append(zr.max((2, 4))); acc[c]["cmin"].append(zr.min((2, 4)))
        else:   # candidate coarser grid than target: upsample by repeat
            g = Rt // RES(p[0]); acc[c]["cmax"].append(zp.repeat(g, 1).repeat(g, 2)); acc[c]["cmin"].append(zp.repeat(g, 1).repeat(g, 2))
zt = np.concatenate(tm_all); imax = zt.max((1, 2)); imin = zt.min((1, 2))
ton = zt > .5 * np.percentile(imax, 99.9); tneg = zt < .5 * np.percentile(imin, .1)
print(f"target {T}: on frac {ton.mean():.5f} ({ton.sum()} spots), neg frac {tneg.mean():.5f}")
res = {}
print("cand  block,n      fwd   offside  rev   breadth  lift  negfwd  negoff | mean z/max at targ-on")
for c, p in cands.items():
    cm = np.concatenate(acc[c]["cmax"]); cn = np.concatenate(acc[c]["cmin"])
    con = cm > .5 * np.percentile(cm.max((1, 2)), 99.9); cng = cn < .5 * np.percentile(cn.min((1, 2)), .1)
    fwd = con[ton].mean(); off = cng[ton].mean(); rev = ton[con].mean() if con.any() else 0; br = con.mean()
    nf = con[tneg].mean() if tneg.any() else 0; no = cng[tneg].mean() if tneg.any() else 0
    mz = float(np.median(cm[ton]) / np.percentile(cm.max((1, 2)), 99.9))
    rm = np.percentile(cm.max((1, 2)), 99.9); c25 = cm > .25 * rm; cpos = cm > 0
    dil = cm.copy(); pad = np.pad(cm, ((0, 0), (1, 1), (1, 1)), constant_values=-1e9)
    for dr in range(3):
        for dc in range(3): dil = np.maximum(dil, pad[:, dr:dr + cm.shape[1], dc:dc + cm.shape[2]])
    d25 = dil > .25 * rm; tl = zt > .25 * np.percentile(imax, 99.9)
    loose = dict(fwd25=float(c25[ton].mean()), rev25=float(ton[c25].mean()), br25=float(c25.mean()), fwd25dil=float(d25[ton].mean()),
                 fwdpos=float(cpos[ton].mean()), brpos=float(cpos.mean()), loose_fwd=float(c25[tl].mean()), loose_rev=float(tl[c25].mean()),
                 tloose_br=float(tl.mean()), negbr=float(cng.mean()), negrev=float(ton[cng].mean()) if cng.any() else 0.0, negrev_loose=float(tl[cng].mean()) if cng.any() else 0.0, corr=float(np.corrcoef(cm.ravel()[::7], zt.ravel()[::7])[0, 1]))
    res[c] = dict(p=p, fwd=float(fwd), off=float(off), rev=float(rev), breadth=float(br), lift=float(fwd / max(br, 1e-9)), negfwd=float(nf), negoff=float(no), medz=mz, **loose)
    print(f"{c}   {str(p):11s} {fwd:5.2f}  {off:5.2f}   {rev:5.2f}  {br:6.4f}  {fwd/max(br,1e-9):5.1f}  {nf:5.2f}  {no:5.2f}  | {mz:+.2f}")
    print("        loose: " + "  ".join(f"{k} {v:.3f}" for k, v in loose.items()))
# ablation at target top-20 and neg-20 images (spot = image argmax / argmin)
top = np.argsort(-imax)[:20]; neg = np.argsort(imin)[:20]
sel = np.concatenate([top, neg]); imgs = np.asarray(X[IDX[sel]]); base = maps(imgs, [T])[T]
bt = base[:20].reshape(20, -1); bn = base[20:].reshape(20, -1); pt = bt.argmax(1); pn = bn.argmin(1)
print("\nablation: median fractional change of target z at its top-20 spots / at its neg-20 spots (all paths)")
for c, p in cands.items():
    a = maps(imgs, [T], kill=p)[T]; at = a[:20].reshape(20, -1); an = a[20:].reshape(20, -1)
    dt = np.median((bt[np.arange(20), pt] - at[np.arange(20), pt]) / bt[np.arange(20), pt])
    dn = np.median((bn[np.arange(20), pn] - an[np.arange(20), pn]) / np.abs(bn[np.arange(20), pn]))
    res[c]["abl_top"] = float(dt); res[c]["abl_neg"] = float(dn)
    print(f"{c}   {str(p):11s} top drop {dt:+.3f}   neg change {dn:+.3f} (positive = neg spot became less negative)")
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", f"judge_{key}.json"), "w"), indent=1)
