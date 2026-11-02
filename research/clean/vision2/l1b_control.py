"""L1b control: random non-family cut that replaces a parent + its children. Rules: VISION-LOG.md L1."""
import json, os, time, resource
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from PIL import Image, ImageDraw
from convnext import *
t0 = time.time(); m = load_convnext(); X, y = load_images(); ST = m.stage_of; NB = 18
H = [m.blocks[b].fc1.weight.shape[0] for b in range(NB)]
rank = np.zeros(len(y), int)
for c in range(1000): ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
ANA = np.where(rank < 20)[0]; SEAL = np.where((rank >= 20) & (rank < 30))[0]; rng = np.random.default_rng(0)
MAP = json.load(open(os.path.join(ROOT, "research/clean/map/network_map_v2.json")))["neurons"]; kids = {}
for k, v in MAP.items(): b, n = map(int, k.split("_")); kids.setdefault(tuple(v["parent"]), []).append((b, n))
FAM = [(15, 1162), (15, 2142), (16, 2886), (13, 78), (10, 1172), (9, 686)]
HANDC = {(15, 1162): (16, 384), (15, 2142): (16, 26), (16, 2886): (17, 2283), (13, 78): (14, 1326), (10, 1172): (11, 1031), (9, 686): (10, 1316)}
CH = {p: [HANDC[p]] + [c for c in kids.get(p, []) if c[0] == p[0] + 1 and c != HANDC[p]] for p in FAM}
def _sub(p):
    st, out = [p], set()
    while st:
        q = st.pop()
        for c in kids.get(q, []): out.add(c); st.append(c)
    return out
RCH = {}
for p in FAM:
    cs0 = CH[p][:5]; sub = _sub(p) | set(CH[p]); b = p[0] + 1
    RCH[p] = [(b, int(j)) for j in rng.choice([j for j in range(H[b]) if (b, j) not in sub], 400, replace=False)]
CH0 = CH; CH = {p: CH0[p][:5] for p in FAM}; CAND = RCH
print({p: len(v) for p, v in CH.items()}, flush=True)
gelu = nn.gelu

def to_block(xs, b):
    x = m.stem_norm(m.stem(xs))
    for i in range(b):
        if i > 0 and ST[i] != ST[i - 1]: x = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](x))
        x = m.blocks[i](x)
    if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
    return x

def run_from(x, b, nrn=None, pn=None, keep_h=False):
    hs = None
    for i in range(b, NB):
        if i > b and ST[i] != ST[i - 1]: x = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](x))
        blk = m.blocks[i]
        if i == b and nrn is not None:
            pre = blk.pre(x); h = gelu(blk.fc1(pre)); h = h * MASK
            hs = gelu(pre @ nrn["u"] + nrn["b0"]); x = x + blk.gamma * (blk.fc2(h) + hs[..., None] * nrn["w"])
        else: x = blk(x)
    lg = m.head(x); return (lg, hs) if keep_h else lg

# one pass: per-image max of every family neuron, on analysis + sealed images, and positional means on random images
need = {}
for p in FAM:
    need.setdefault(p[0], set()).add(p[1])
    for c in CH[p] + CAND[p]: need.setdefault(c[0], set()).add(c[1])
need = {b: sorted(v) for b, v in need.items()}; IMGS = np.concatenate([ANA, SEAL]); MXV = {(b, n): [] for b, ns in need.items() for n in ns}; MEAN = {k: [] for k in MXV}
RND = set(rng.choice(len(IMGS), 2000, replace=False).tolist())
for i in range(0, len(IMGS), 100):
    x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[IMGS[i:i + 100]])))))
    for b in range(max(need) + 1):
        if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
        if b in need:
            h = np.array(gelu(m.blocks[b].fc1(m.blocks[b].pre(x)))[..., need[b]]); hm = h.reshape(h.shape[0], -1, h.shape[-1])
            sel = [j for j in range(h.shape[0]) if i + j in RND]
            for q, n in enumerate(need[b]):
                MXV[(b, n)].append(hm[:, :, q].max(1))
                if sel: MEAN[(b, n)].append(hm[sel, :, q].mean())
        x = m.blocks[b](x)
MXV = {k: np.concatenate(v) for k, v in MXV.items()}; MEAN = {k: float(np.mean(v)) for k, v in MEAN.items()}
print("scan", f"{time.time() - t0:.0f}s", flush=True)
OUT = {}; G = lambda b: np.array(m.blocks[b].gamma)
for p in FAM:
    pb, pn = p; cs = CH[p]
    if len(cs) > 5: cs = cs[:5]
    pick = []
    for c in cs:
        ok = [r for r in CAND[p] if r not in pick and MEAN[c] / 2 < MEAN.get(r, -1) < MEAN[c] * 2]
        pick.append(ok[0] if ok else [r for r in CAND[p] if r not in pick][0])
    cs = pick
    if False:
        en = [MXV[c].mean() * np.linalg.norm(np.array(m.blocks[c[0]].fc2.weight)[:, c[1]]) for c in cs]; cs = [cs[0]] + [cs[1:][i] for i in np.argsort(en[1:])[::-1][:4]]
    sc = MXV[p] / MXV[p].max() + np.mean([MXV[c] / max(MXV[c].max(), 1e-6) for c in cs], 0)
    na = len(ANA); a_sc, s_sc = sc[:na], sc[na:]
    TR = ANA[np.r_[np.argsort(-a_sc)[:2000], rng.choice(np.argsort(-a_sc)[2000:], 1000, replace=False)]]
    EVs = SEAL[np.r_[np.argsort(-s_sc)[:500], rng.choice(np.argsort(-s_sc)[500:], 500, replace=False)]]
    cache = lambda I: mx.concatenate([to_block(mx.array(norm_img(np.asarray(X[np.sort(I)[k:k + 50]]))), pb) for k in range(0, len(I), 50)])
    TRs, EVsrt = np.sort(TR), np.sort(EVs); XT, XE = cache(TR), cache(EVs); mx.eval(XT, XE)
    MASK = mx.ones((H[pb],)); 
    TT = mx.concatenate([run_from(XT[k:k + 50], pb) for k in range(0, len(TR), 50)]); TE = mx.concatenate([run_from(XE[k:k + 50], pb) for k in range(0, len(EVs), 50)]); mx.eval(TT, TE)
    # edit children (mean replacement) in block cb
    cb = pb + 1; F2c = m.blocks[cb].fc2; W2 = np.array(F2c.weight); b2 = np.array(F2c.bias)
    for c in cs: b2 = b2 + W2[:, c[1]] * MEAN[c]; W2[:, c[1]] = 0
    f = nn.Linear(W2.shape[1], W2.shape[0]); f.weight = mx.array(W2); f.bias = mx.array(b2); m.blocks[cb].fc2 = f
    def klv(lg, T):
        lp = lg - mx.logsumexp(lg, axis=1, keepdims=True); tp = T - mx.logsumexp(T, axis=1, keepdims=True); return (mx.exp(tp) * (tp - lp)).sum(1)
    def evalkl(nrn=None, cut_parent=False):
        global MASK
        if cut_parent:   # parent + children cut: parent replaced by its mean via a constant neuron
            MASK = mx.array((np.arange(H[pb]) != pn).astype(np.float32)); nr = {"u": mx.zeros((XE.shape[-1],)), "b0": mx.array(10.0), "w": mx.array(np.array(m.blocks[pb].fc2.weight)[:, pn] * MEAN[p] / 10.0)}
            r = mx.concatenate([klv(run_from(XE[k:k + 50], pb, nr), TE[k:k + 50]) for k in range(0, len(EVs), 50)]); MASK = mx.ones((H[pb],)); return np.array(r)
        if nrn is None: return np.array(mx.concatenate([klv(run_from(XE[k:k + 50], pb), TE[k:k + 50]) for k in range(0, len(EVs), 50)]))
        MASK = mx.array((np.arange(H[pb]) != pn).astype(np.float32))
        r = mx.concatenate([klv(run_from(XE[k:k + 50], pb, nrn), TE[k:k + 50]) for k in range(0, len(EVs), 50)]); return np.array(r)
    k_naive = evalkl(); k_both = evalkl(cut_parent=True)
    W1 = np.array(m.blocks[pb].fc1.weight); B1 = np.array(m.blocks[pb].fc1.bias); W2p = np.array(m.blocks[pb].fc2.weight)
    nrn = {"u": mx.array(W1[pn]), "b0": mx.array(B1[pn]), "w": mx.array(W2p[:, pn])}
    MASK = mx.array((np.arange(H[pb]) != pn).astype(np.float32))
    def loss(P, xb, tb): return klv(run_from(xb, pb, P), tb).mean()
    lvg = mx.value_and_grad(loss); opt = optim.Adam(learning_rate=3e-4); ls = []
    for ep in range(3):
        perm = rng.permutation(len(TR))
        for k in range(0, len(TR), 50):
            ix = mx.array(perm[k:k + 50]); l, g = lvg(nrn, XT[ix], TT[ix]); opt.update(nrn, g); mx.eval(nrn, opt.state); ls.append(float(l))
    k_learn = evalkl(nrn)
    # what did it become? activations on held-out positions
    hp_, hc_, hs_ = [], [], []
    for k in range(0, len(EVs), 50):
        xb = XE[k:k + 50]; pre = m.blocks[pb].pre(xb); hp_.append(np.array(gelu(m.blocks[pb].fc1(pre))[..., pn]).reshape(-1))
        hs_.append(np.array(gelu(pre @ nrn["u"] + nrn["b0"])).reshape(-1))
        x2 = m.blocks[pb](xb); F2c_e = m.blocks[cb].fc2; m.blocks[cb].fc2 = F2c
        hc_.append(np.array(gelu(m.blocks[cb].fc1(m.blocks[cb].pre(x2)))[..., [c[1] for c in cs]]).reshape(-1, len(cs))); m.blocks[cb].fc2 = F2c_e
    hp_, hs_, hc_ = np.concatenate(hp_), np.concatenate(hs_), np.concatenate(hc_); hcs = hc_.sum(1)
    A = np.c_[hp_, np.ones(len(hp_))]; res_ = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    pc = float(np.corrcoef(res_(hs_), res_(hcs))[0, 1]); pc_each = [float(np.corrcoef(res_(hs_), res_(hc_[:, j]))[0, 1]) for j in range(len(cs))]
    u0 = W1[pn]; u1 = np.array(nrn["u"]); w0 = W2p[:, pn]; w1 = np.array(nrn["w"])
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    wc_dirs = [np.array(m.blocks[c[0]].fc2.weight)[:, c[1]] * G(c[0]) for c in cs]; dw = w1 * G(pb) - w0 * G(pb)
    o = {"parent": list(p), "children": [list(c) for c in cs], "kl_naive_cut": float(k_naive.mean()), "kl_cut_both": float(k_both.mean()), "kl_learned": float(k_learn.mean()),
         "share_won_back": float((k_naive.mean() - k_learn.mean()) / k_naive.mean()), "share_top500": float((k_naive[:0].sum() if False else 0)),
         "loss_first_last": [float(np.mean(ls[:10])), float(np.mean(ls[-10:]))], "corr_learned_parent": float(np.corrcoef(hs_, hp_)[0, 1]), "corr_learned_children_sum": float(np.corrcoef(hs_, hcs)[0, 1]),
         "partial_corr_children_given_parent": pc, "partial_corr_each_child": pc_each, "cos_read_vs_parent": cos(u0, u1), "cos_write_vs_parent": cos(w0, w1),
         "write_norm_ratio": float(np.linalg.norm(w1) / np.linalg.norm(w0)), "write_change_cos_children": [cos(dw, wc) for wc in wc_dirs], "bias_change": float(np.array(nrn["b0"]) - B1[pn])}
    rr = np.random.default_rng(1); d = (k_naive - k_learn) / k_naive.mean(); bs = [d[rr.integers(0, len(d), len(d))].mean() for _ in range(1000)]; o["share_ci"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    OUT[f"{pb}_{pn}"] = o; print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}), f"{time.time() - t0:.0f}s maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f}GB", flush=True)
    np.savez(os.path.join(HERE, "figs", "l1", f"ctrl_acts_{pb}_{pn}.npz"), hp=hp_, hs=hs_, hc=hc_, ev=EVsrt, u=u1, b0=np.array(nrn["b0"]), w=w1)
    m.blocks[cb].fc2 = F2c; del XT, XE, TT, TE
json.dump(OUT, open(os.path.join(RES, "vision-l1b-control.json"), "w"), indent=1)
