"""Harness for the closed-form nudge (VISION-LOG: CLOSED-FORM NUDGE). from n_harness import *; f = Fam("13_78")."""
import json, os, time
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
t0 = time.time(); m = load_convnext(); X, y = load_images(); ST = m.stage_of; NB = 18
H = [m.blocks[b].fc1.weight.shape[0] for b in range(NB)]
rank = np.zeros(len(y), int)
for c in range(1000): ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
ANA = np.where(rank < 20)[0]
R1 = json.load(open(os.path.join(RES, "vision-l1-learn.json"))); CACHE = os.path.join(HF, "imagenet", "l1cache"); os.makedirs(CACHE, exist_ok=True)
gelu = nn.gelu; F2 = [m.blocks[b].fc2 for b in range(NB)]
GM = [np.array(m.blocks[b].gamma, np.float64) for b in range(NB)]

def to_block(xs, b):
    x = m.stem_norm(m.stem(xs))
    for i in range(b):
        if i > 0 and ST[i] != ST[i - 1]: x = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](x))
        x = m.blocks[i](x)
    if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
    return x

def batches(n, k=50): return [(i, min(i + k, n)) for i in range(0, n, k)]

class Fam:
    def __init__(self, key, ntr=3000):
        o = R1[key]; self.key = key; self.p = tuple(o["parent"]); self.cs = [tuple(c) for c in o["children"]]; pb, pn = self.p; self.pb, self.pn, self.cb = pb, pn, pb + 1
        d = np.load(os.path.join(HERE, "figs", "l1", f"acts_{key}.npz")); self.ev = d["ev"]; self.L = {"u": d["u"], "b0": float(d["b0"]), "w": d["w"]}
        cf = os.path.join(CACHE, f"{key}.npz")
        if os.path.exists(cf):
            z = np.load(cf); self.tr = z["tr"]; self.XT = mx.array(z["XT"]); self.XE = mx.array(z["XE"]); self.mu = z["mu"]
        else:
            rng = np.random.default_rng(11); pool = np.sort(rng.choice(ANA, 8000, replace=False)); sc = []; hs = []
            for a, b in batches(len(pool), 100):
                x = to_block(mx.array(norm_img(np.asarray(X[pool[a:b]]))), pb); hp = gelu(m.blocks[pb].fc1(m.blocks[pb].pre(x))[..., pn])
                x2 = m.blocks[pb](x); hc = gelu(m.blocks[self.cb].fc1(m.blocks[self.cb].pre(x2))[..., mx.array([c[1] for c in self.cs])])
                sc.append(np.c_[np.array(hp.reshape(hp.shape[0], -1).max(1)), np.array(hc.reshape(hc.shape[0], -1, len(self.cs)).max(1))]); hs.append(np.array(hc.reshape(-1, len(self.cs)).mean(0)) * (b - a))
            sc = np.concatenate(sc); s = (sc / np.maximum(sc.max(0), 1e-6)); s = s[:, 0] + s[:, 1:].mean(1); o_ = np.argsort(-s)
            self.tr = np.sort(pool[np.r_[o_[:2000], rng.choice(o_[2000:], ntr - 2000, replace=False)]]); self.mu = np.sum(hs, 0) / len(pool)
            self.XT = mx.concatenate([to_block(mx.array(norm_img(np.asarray(X[self.tr[a:b]]))), pb) for a, b in batches(len(self.tr))])
            self.XE = mx.concatenate([to_block(mx.array(norm_img(np.asarray(X[self.ev[a:b]]))), pb) for a, b in batches(len(self.ev))])
            np.savez(cf, tr=self.tr, XT=np.array(self.XT).astype(np.float16), XE=np.array(self.XE).astype(np.float16), mu=self.mu)
            self.XT, self.XE = mx.array(np.array(self.XT).astype(np.float16)), mx.array(np.array(self.XE).astype(np.float16))
        self.XT = self.XT.astype(mx.float32); self.XE = self.XE.astype(mx.float32)
        self.TE = self.run(self.XE); mx.eval(self.TE); self._TT = None
        # child cut (mean replacement) as an fc2 edit of block cb
        W2 = np.array(F2[self.cb].weight); b2 = np.array(F2[self.cb].bias)
        for c, mu in zip(self.cs, self.mu): b2 = b2 + W2[:, c[1]] * mu; W2[:, c[1]] = 0
        self.fc2cut = nn.Linear(W2.shape[1], W2.shape[0]); self.fc2cut.weight = mx.array(W2); self.fc2cut.bias = mx.array(b2)
        self.W1 = np.array(m.blocks[pb].fc1.weight); self.B1 = np.array(m.blocks[pb].fc1.bias); self.W2p = np.array(F2[pb].weight)
        self.parent = {"u": self.W1[pn], "b0": float(self.B1[pn]), "w": self.W2p[:, pn]}
        self.wc = [np.array(F2[self.cb].weight)[:, c[1]] for c in self.cs]          # child writes (model units, before gamma of block cb)
        self.k_naive = self.evalkl(None)

    @property
    def TT(self):
        if self._TT is None: self._TT = self.run(self.XT); mx.eval(self._TT)
        return self._TT

    def run(self, x, nrn=None, dbias=None, cut=False):
        """logits from block pb input. nrn replaces the parent (dict u, b0, w; numpy ok). dbias: extra vector on block pb's fc2 bias."""
        pb = self.pb
        if cut: m.blocks[self.cb].fc2 = self.fc2cut
        out = []
        for a, b in batches(x.shape[0]):
            h = x[a:b]
            for i in range(pb, NB):
                if i > pb and ST[i] != ST[i - 1]: h = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](h))
                blk = m.blocks[i]
                if i == pb and (nrn is not None or dbias is not None):
                    pre = blk.pre(h); z = blk.fc1(pre); hh = gelu(z); mo = blk.fc2(hh)
                    if nrn is not None:
                        mo = mo - hh[..., self.pn:self.pn + 1] * mx.array(self.W2p[:, self.pn]) + gelu(pre @ mx.array(nrn["u"]) + nrn["b0"])[..., None] * mx.array(nrn["w"])
                    if dbias is not None: mo = mo + mx.array(dbias)
                    h = h + blk.gamma * mo
                else: h = blk(h)
            out.append(m.head(h))
        m.blocks[self.cb].fc2 = F2[self.cb]; return mx.concatenate(out)

    def kl(self, lg, T):
        lp = lg - mx.logsumexp(lg, axis=1, keepdims=True); tp = T - mx.logsumexp(T, axis=1, keepdims=True); return np.array((mx.exp(tp) * (tp - lp)).sum(1))

    def evalkl(self, nrn, dbias=None):
        return self.kl(self.run(self.XE, nrn, dbias, cut=True), self.TE)

    def share(self, nrn, dbias=None):
        k = self.evalkl(nrn, dbias); return float((self.k_naive.mean() - k.mean()) / self.k_naive.mean())

    def acts(self, x, nmax=None):
        """per position: parent input (pre), z_p, children z_c (intact), as float64 numpy."""
        P, ZP, ZC = [], [], []
        for a, b in batches(x.shape[0] if nmax is None else min(nmax, x.shape[0])):
            h = x[a:b]; pre = m.blocks[self.pb].pre(h); z = m.blocks[self.pb].fc1(pre)[..., self.pn]; h2 = m.blocks[self.pb](h)
            zc = m.blocks[self.cb].fc1(m.blocks[self.cb].pre(h2))[..., mx.array([c[1] for c in self.cs])]
            P.append(np.array(pre.reshape(-1, pre.shape[-1]))); ZP.append(np.array(z.reshape(-1))); ZC.append(np.array(zc.reshape(-1, len(self.cs))))
        return np.concatenate(P).astype(np.float64), np.concatenate(ZP).astype(np.float64), np.concatenate(ZC).astype(np.float64)

FAMS = list(R1)
