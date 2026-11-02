"""N7: one-time cache of 12 families (fast-iteration set)."""

import json, os, sys, time
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from convnext import *
t0 = time.time(); m = load_convnext(); X, y = load_images(); ST = m.stage_of; NB = 18
H = [m.blocks[b].fc1.weight.shape[0] for b in range(NB)]; GM = [np.array(m.blocks[b].gamma, np.float64) for b in range(NB)]; F2 = [m.blocks[b].fc2 for b in range(NB)]
rank = np.zeros(len(y), int)
for c in range(1000): ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
ANA = np.where(rank < 20)[0]; SEAL = np.where((rank >= 20) & (rank < 30))[0]; rng = np.random.default_rng(21); gelu = nn.gelu
MAP = json.load(open(os.path.join(ROOT, "research/clean/map/network_map_v2.json")))["neurons"]; kids = {}
for k, v in MAP.items(): b, n = map(int, k.split("_")); kids.setdefault(tuple(v["parent"]), []).append((b, n))
HANDC = {(15, 1162): (16, 384), (15, 2142): (16, 26), (16, 2886): (17, 2283), (13, 78): (14, 1326), (10, 1172): (11, 1031), (9, 686): (10, 1316)}
FAMS = {p: [c] + [q for q in kids.get(p, []) if q[0] == p[0] + 1 and q != c] for p, c in HANDC.items()}
for blks, nfam in (((9, 10, 11, 12, 13), 3), ((15, 16), 3)):
    cand = [p for p in kids if p[0] in blks and p not in FAMS and any(q[0] == p[0] + 1 for q in kids[p])]
    for j in rng.choice(len(cand), nfam, replace=False): p = cand[j]; FAMS[p] = [q for q in kids[p] if q[0] == p[0] + 1]
OUTF = None
# ---- one scan: per-image max of every family neuron over analysis + sealed images, and positional means (random images)
need = {}
for p, cs in FAMS.items():
    for q in [p] + cs[:12]: need.setdefault(q[0], set()).add(q[1])
need = {b: sorted(v) for b, v in need.items()}; IM = np.concatenate([ANA, SEAL]); MXV = {(b, n): [] for b in need for n in need[b]}; SUM = {k: 0.0 for k in MXV}; cnt = 0
RND = set(rng.choice(len(IM), 2000, replace=False).tolist())
for i in range(0, len(IM), 100):
    x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[IM[i:i + 100]]))))); sel = [j for j in range(min(100, len(IM) - i)) if i + j in RND]
    for b in range(max(need) + 1):
        if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
        if b in need:
            h = np.array(gelu(m.blocks[b].fc1(m.blocks[b].pre(x)))[..., need[b]]); hm = h.reshape(h.shape[0], -1, h.shape[-1])
            for q, n in enumerate(need[b]):
                MXV[(b, n)].append(hm[:, :, q].max(1))
                if sel: SUM[(b, n)] += float(hm[sel, :, q].mean(1).sum())
        x = m.blocks[b](x)
    cnt += len(sel)
MXV = {k: np.concatenate(v) for k, v in MXV.items()}; MEAN = {k: v / cnt for k, v in SUM.items()}; print("scan", f"{time.time() - t0:.0f}s", flush=True)

def to_block(xs, b):
    x = m.stem_norm(m.stem(xs))
    for i in range(b):
        if i > 0 and ST[i] != ST[i - 1]: x = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](x))
        x = m.blocks[i](x)
    if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
    return x
def batches(n, k=50): return [(i, min(i + k, n)) for i in range(0, n, k)]
def gelu_d(z): return 0.5 * (1 + mx.erf(z / 2 ** .5)) + z * mx.exp(-z * z / 2) / (2 * np.pi) ** .5

class Fam:
    def __init__(self, p, cs):
        self.p = p; pb, pn = p; self.pb, self.pn, self.cb = pb, pn, pb + 1
        cs = [c for c in cs if c in MEAN]
        if len(cs) > 5: cs = [cs[0]] + sorted(cs[1:], key=lambda c: -MEAN[c] * np.linalg.norm(np.array(F2[c[0]].weight)[:, c[1]]))[:4] if p in HANDC else sorted(cs, key=lambda c: -MEAN[c] * np.linalg.norm(np.array(F2[c[0]].weight)[:, c[1]]))[:5]
        self.cs = cs; sc = MXV[p] / max(MXV[p].max(), 1e-6) + np.mean([MXV[c] / max(MXV[c].max(), 1e-6) for c in cs], 0); na = len(ANA)
        a_, s_ = np.argsort(-sc[:na]), np.argsort(-sc[na:])
        self.tr = np.sort(ANA[np.r_[a_[:1000], rng.choice(a_[1000:], 500, replace=False)]]); self.ev = np.sort(SEAL[np.r_[s_[:250], rng.choice(s_[250:], 250, replace=False)]])
        cache = lambda I: mx.concatenate([to_block(mx.array(norm_img(np.asarray(X[I[a:b]]))), pb) for a, b in batches(len(I))])
        self.XT, self.XE = cache(self.tr), cache(self.ev); mx.eval(self.XT, self.XE)
        self.TT, self.TE = self.run(self.XT), self.run(self.XE); mx.eval(self.TT, self.TE)
        W2 = np.array(F2[self.cb].weight); b2 = np.array(F2[self.cb].bias)
        for c in cs: b2 = b2 + W2[:, c[1]] * MEAN[c]; W2[:, c[1]] = 0
        self.fc2cut = nn.Linear(W2.shape[1], W2.shape[0]); self.fc2cut.weight = mx.array(W2); self.fc2cut.bias = mx.array(b2)
        W1 = np.array(m.blocks[pb].fc1.weight); B1 = np.array(m.blocks[pb].fc1.bias); self.W2p = np.array(F2[pb].weight)
        self.parent = {"u": W1[pn].astype(np.float64), "b0": float(B1[pn]), "w": self.W2p[:, pn].astype(np.float64)}
        self.mask = mx.array((np.arange(H[pb]) != pn).astype(np.float32))
        self.k_naive = self.kl(self.run(self.XE, cut=True), self.TE)
    def run(self, x, nrn=None, cut=False):
        pb = self.pb
        if cut: m.blocks[self.cb].fc2 = self.fc2cut
        out = []
        for a, b in batches(x.shape[0]):
            h = x[a:b]
            for i in range(pb, NB):
                if i > pb and ST[i] != ST[i - 1]: h = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](h))
                blk = m.blocks[i]
                if i == pb and nrn is not None:
                    pre = blk.pre(h); hh = gelu(blk.fc1(pre)) * self.mask
                    h = h + blk.gamma * (blk.fc2(hh) + gelu(pre @ mx.array(np.asarray(nrn["u"], np.float32)) + nrn["b0"])[..., None] * mx.array(np.asarray(nrn["w"], np.float32)))
                else: h = blk(h)
            out.append(m.head(h))
        m.blocks[self.cb].fc2 = F2[self.cb]; return mx.concatenate(out)
    def kl(self, lg, T):
        lp = lg - mx.logsumexp(lg, axis=1, keepdims=True); tp = T - mx.logsumexp(T, axis=1, keepdims=True); return np.array((mx.exp(tp) * (tp - lp)).sum(1))
    def share(self, nrn):
        k = self.kl(self.run(self.XE, nrn, cut=True), self.TE); return float((self.k_naive.mean() - k.mean()) / self.k_naive.mean())
    def suffix(self, h):
        for i in range(self.cb, NB):
            if ST[i] != ST[i - 1]: h = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](h))
            h = m.blocks[i](h)
        return m.head(h)

def train(f, epochs=3, lr=3e-4):
    nrn = {k: mx.array(np.asarray(v, np.float32)) for k, v in f.parent.items()}; m.blocks[f.cb].fc2 = f.fc2cut; blk = m.blocks[f.pb]
    def loss(P, xb, tb):
        pre = blk.pre(xb); hh = gelu(blk.fc1(pre)) * f.mask; s1 = xb + blk.gamma * (blk.fc2(hh) + gelu(pre @ P["u"] + P["b0"])[..., None] * P["w"])
        lg = f.suffix(s1); lp = lg - mx.logsumexp(lg, axis=1, keepdims=True); tp = tb - mx.logsumexp(tb, axis=1, keepdims=True); return (mx.exp(tp) * (tp - lp)).sum(1).mean()
    lvg = mx.value_and_grad(loss); opt = optim.Adam(learning_rate=lr)
    for ep in range(epochs):
        perm = rng.permutation(f.XT.shape[0])
        for a in range(0, len(perm), 50):
            ix = mx.array(perm[a:a + 50]); l, g = lvg(nrn, f.XT[ix], f.TT[ix]); opt.update(nrn, g); mx.eval(nrn, opt.state)
    m.blocks[f.cb].fc2 = F2[f.cb]; return {"u": np.array(nrn["u"], np.float64), "b0": float(nrn["b0"].item()), "w": np.array(nrn["w"], np.float64)}

def gn_stats(f, th, X_, T_, S=4, parts=("u", "b", "w")):
    C = len(th["u"]); gam = mx.array(GM[f.pb].astype(np.float32)); m.blocks[f.cb].fc2 = f.fc2cut; blk = m.blocks[f.pb]
    u = mx.array(th["u"].astype(np.float32)); w = mx.array(th["w"].astype(np.float32)); A = None; bv = None
    for a, b in batches(X_.shape[0], 25):
        x = X_[a:b]; pre = blk.pre(x); hh = gelu(blk.fc1(pre)) * f.mask; z = pre @ u + th["b0"]; h = gelu(z); s1 = x + blk.gamma * (blk.fc2(hh) + h[..., None] * w)
        lg = f.suffix(s1); p = mx.softmax(lg, axis=1); ps = mx.softmax(T_[a:b], axis=1); pn_ = np.array(p, np.float64); pn_ /= pn_.sum(1, keepdims=True)
        vs = [p - ps] + [mx.eye(1000)[mx.array([rng.choice(1000, p=q) for q in pn_])] - p for _ in range(S)]; dz = gelu_d(z); n_ = b - a
        for k, v in enumerate(vs):
            _, (g1,) = mx.vjp(f.suffix, [s1], [v]); gwh = g1 @ (gam * w); cols = []
            if "u" in parts: cols.append(mx.einsum("np,npk->nk", (gwh * dz).reshape(n_, -1), pre.reshape(n_, -1, pre.shape[-1])))
            if "b" in parts: cols.append((gwh * dz).reshape(n_, -1).sum(1, keepdims=True))
            if "w" in parts: cols.append((h.reshape(n_, -1, 1) * g1.reshape(n_, -1, g1.shape[-1])).sum(1) * gam)
            phi = np.array(mx.concatenate(cols, axis=1), np.float64)
            if A is None: A = np.zeros((phi.shape[1],) * 2); bv = np.zeros(phi.shape[1])
            if k == 0: bv += phi.sum(0)
            else: A += phi.T @ phi / S
    m.blocks[f.cb].fc2 = F2[f.cb]; return A, bv

def apply(th, d, parts, C):
    t = {"u": th["u"].copy(), "b0": th["b0"], "w": th["w"].copy()}; i = 0
    if "u" in parts: t["u"] = t["u"] + d[i:i + C]; i += C
    if "b" in parts: t["b0"] = float(t["b0"] + d[i]); i += 1
    if "w" in parts: t["w"] = t["w"] + d[i:i + C]
    return t

def gn(f, th, parts=("u", "b", "w"), nit=5, log=None):
    C = len(th["u"]); XG, TG, XV, TV = f.XT[::3 * 0 + 1][:2000], f.TT[:2000], f.XT[2000:], f.TT[2000:]
    idx = np.arange(f.XT.shape[0]); g_ix, v_ix = mx.array(idx[idx % 3 != 0]), mx.array(idx[idx % 3 == 0])
    XG, TG, XV, TV = f.XT[g_ix], f.TT[g_ix], f.XT[v_ix], f.TT[v_ix]
    kv0 = f.kl(f.run(XV, th, cut=True), TV).mean(); hist = []
    for it in range(nit):
        A, bv = gn_stats(f, th, XG, TG, parts=parts); dg = np.diag(A); best = (kv0, None, None)
        for lm in (1e-2, 1e-1, 1, 10, 100):
            d = -np.linalg.solve(A + lm * np.diag(dg + 1e-9 * dg.max()), bv)
            for al in (.25, .5, 1.0):
                cand = apply(th, al * d, parts, C); kv = f.kl(f.run(XV, cand, cut=True), TV).mean()
                if kv < best[0]: best = (kv, cand, (lm, al))
        if best[1] is None: hist.append(None); break
        th, kv0 = best[1], best[0]; hist.append((round(f.share(th), 3), best[2]))
    return th, hist


CD = os.path.join(HF, "imagenet", "n7cache"); os.makedirs(CD, exist_ok=True)
for p, cs in FAMS.items():
    fn = os.path.join(CD, f"{p[0]}_{p[1]}.npz")
    if os.path.exists(fn): continue
    f = Fam(p, cs); L = train(f, epochs=4)
    np.savez(fn, p=np.array(p), cs=np.array(f.cs), tr=f.tr, ev=f.ev, XT=np.array(f.XT).astype(np.float16), XE=np.array(f.XE).astype(np.float16), TT=np.array(f.TT), TE=np.array(f.TE),
             mu=np.array([MEAN[c] for c in f.cs]), Lu=L["u"], Lb=L["b0"], Lw=L["w"])
    print(p, "children", f.cs, "kl_naive %.5f trained share %.3f" % (f.k_naive.mean(), f.share(L)), f"{time.time() - t0:.0f}s", flush=True)
    del f
