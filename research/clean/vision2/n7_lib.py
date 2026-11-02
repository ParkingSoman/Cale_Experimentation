"""Fast-iteration library for the closed-form nudge. from n7_lib import *; f = load("15_1162")."""
import json, os, sys, time, glob
import numpy as np, mlx.core as mx, mlx.nn as nn
from convnext import *
t0 = time.time(); m = load_convnext(); ST = m.stage_of; NB = 18
H = [m.blocks[b].fc1.weight.shape[0] for b in range(NB)]; GM = [np.array(m.blocks[b].gamma, np.float64) for b in range(NB)]; F2 = [m.blocks[b].fc2 for b in range(NB)]
gelu = nn.gelu; rng = np.random.default_rng(7); CD = os.path.join(HF, "imagenet", "n7cache")
KEYS = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CD, "*.npz")) if not os.path.basename(p).startswith("gn_"))
def batches(n, k=50): return [(i, min(i + k, n)) for i in range(0, n, k)]
def gelu_d(z): return 0.5 * (1 + mx.erf(z / 2 ** .5)) + z * mx.exp(-z * z / 2) / (2 * np.pi) ** .5
gl = lambda z: z * 0.5 * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))

class Fam:
    def __init__(self, key):
        z = np.load(os.path.join(CD, key + ".npz")); self.key = key
        self.p = tuple(int(v) for v in z["p"]); self.cs = [tuple(int(v) for v in c) for c in z["cs"]]; pb, pn = self.p; self.pb, self.pn, self.cb = pb, pn, pb + 1
        self.tr, self.ev = z["tr"], z["ev"]; self.XT = mx.array(z["XT"]).astype(mx.float32); self.XE = mx.array(z["XE"]).astype(mx.float32); self.TT = mx.array(z["TT"]); self.TE = mx.array(z["TE"])
        self.mu = z["mu"]; self.L = {"u": z["Lu"], "b0": float(z["Lb"]), "w": z["Lw"]}
        W2 = np.array(F2[self.cb].weight); b2 = np.array(F2[self.cb].bias)
        for c, mu in zip(self.cs, self.mu): b2 = b2 + W2[:, c[1]] * mu; W2[:, c[1]] = 0
        self.fc2cut = nn.Linear(W2.shape[1], W2.shape[0]); self.fc2cut.weight = mx.array(W2); self.fc2cut.bias = mx.array(b2)
        W1 = np.array(m.blocks[pb].fc1.weight); B1 = np.array(m.blocks[pb].fc1.bias)
        self.parent = {"u": W1[pn].astype(np.float64), "b0": float(B1[pn]), "w": np.array(F2[pb].weight)[:, pn].astype(np.float64)}
        self.wc = np.stack([np.array(F2[self.cb].weight)[:, c[1]] for c in self.cs], 1).astype(np.float64)        # model units of block cb
        self.mask = mx.array((np.arange(H[pb]) != pn).astype(np.float32))
        idx = np.arange(self.XT.shape[0]); self.gix, self.vix = mx.array(idx[idx % 3 != 0]), mx.array(idx[idx % 3 == 0])
        self.k_naive = self.kl(self.run(self.XE, cut=True), self.TE)
    def block_pb(self, x, nrn):
        blk = m.blocks[self.pb]; pre = blk.pre(x); hh = gelu(blk.fc1(pre)) * self.mask
        return x + blk.gamma * (blk.fc2(hh) + gelu(pre @ mx.array(np.asarray(nrn["u"], np.float32)) + nrn["b0"])[..., None] * mx.array(np.asarray(nrn["w"], np.float32))), pre
    def suffix(self, h):
        for i in range(self.cb, NB):
            if ST[i] != ST[i - 1]: h = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](h))
            h = m.blocks[i](h)
        return m.head(h)
    def run(self, x, nrn=None, cut=False):
        if cut: m.blocks[self.cb].fc2 = self.fc2cut
        out = []
        for a, b in batches(x.shape[0]):
            h = x[a:b]
            if nrn is None: h = m.blocks[self.pb](h)
            else: h, _ = self.block_pb(h, nrn)
            out.append(self.suffix(h))
        m.blocks[self.cb].fc2 = F2[self.cb]; return mx.concatenate(out)
    def kl(self, lg, T):
        lp = lg - mx.logsumexp(lg, axis=1, keepdims=True); tp = T - mx.logsumexp(T, axis=1, keepdims=True); return np.array((mx.exp(tp) * (tp - lp)).sum(1))
    def share(self, nrn, on="eval"):
        X_, T_, k0 = (self.XE, self.TE, self.k_naive) if on == "eval" else (self.XT[self.vix], self.TT[self.vix], None)
        k = self.kl(self.run(X_, nrn, cut=True), T_)
        if k0 is None: k0 = self.kl(self.run(X_, cut=True), T_)
        return float((k0.mean() - k.mean()) / k0.mean())
    def trainkl(self, nrn, which="v"):
        ix = self.vix if which == "v" else self.gix; return float(self.kl(self.run(self.XT[ix], nrn, cut=True), self.TT[ix]).mean())

def load(key): return Fam(key)
cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def gn_stats(f, th, ix, S=4, parts=("u", "b", "w")):
    """Gauss-Newton pieces for KL(teacher || student) at theta: gradient b and Fisher curvature A, over train rows ix."""
    gam = mx.array(GM[f.pb].astype(np.float32)); m.blocks[f.cb].fc2 = f.fc2cut; blk = m.blocks[f.pb]
    u = mx.array(np.asarray(th["u"], np.float32)); w = mx.array(np.asarray(th["w"], np.float32)); A = None; bv = None; X_, T_ = f.XT[ix], f.TT[ix]
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

def apply_d(th, d, parts, C):
    t = {"u": np.array(th["u"], np.float64).copy(), "b0": float(th["b0"]), "w": np.array(th["w"], np.float64).copy()}; i = 0
    if "u" in parts: t["u"] += d[i:i + C]; i += C
    if "b" in parts: t["b0"] += float(d[i]); i += 1
    if "w" in parts: t["w"] += d[i:i + C]
    return t

def gn(f, th, parts=("u", "b", "w"), nit=3, lms=(1e-2, 1e-1, 1, 10, 100), als=(.25, .5, 1.0), S=4, verbose=False):
    C = len(th["u"]); kv0 = f.trainkl(th); hist = []
    for it in range(nit):
        A, bv = gn_stats(f, th, f.gix, S=S, parts=parts); dg = np.diag(A); best = (kv0, None, None)
        for lm in lms:
            d = -np.linalg.solve(A + lm * np.diag(dg + 1e-9 * dg.max()), bv)
            for al in als:
                cand = apply_d(th, al * d, parts, C); kv = f.trainkl(cand)
                if kv < best[0]: best = (kv, cand, (lm, al))
        if best[1] is None: break
        th, kv0 = best[1], best[0]; hist.append(best[2])
        if verbose: print("   it", it, best[2], "eval share %.3f" % f.share(th), flush=True)
    return th, hist

import mlx.optimizers as optim
def train(f, epochs=4, lr=3e-4, ix=None, init=None, seed=0):
    r = np.random.default_rng(seed); ix = np.arange(f.XT.shape[0]) if ix is None else np.asarray(ix)
    nrn = {k: mx.array(np.asarray(v, np.float32)) for k, v in (init or f.parent).items()}; m.blocks[f.cb].fc2 = f.fc2cut; blk = m.blocks[f.pb]
    def loss(P, xb, tb):
        pre = blk.pre(xb); hh = gelu(blk.fc1(pre)) * f.mask; s1 = xb + blk.gamma * (blk.fc2(hh) + gelu(pre @ P["u"] + P["b0"])[..., None] * P["w"])
        lg = f.suffix(s1); lp = lg - mx.logsumexp(lg, axis=1, keepdims=True); tp = tb - mx.logsumexp(tb, axis=1, keepdims=True); return (mx.exp(tp) * (tp - lp)).sum(1).mean()
    lvg = mx.value_and_grad(loss); opt = optim.Adam(learning_rate=lr)
    for ep in range(epochs):
        perm = r.permutation(ix)
        for a in range(0, len(perm), 50):
            j = mx.array(np.sort(perm[a:a + 50])); l, g = lvg(nrn, f.XT[j], f.TT[j]); opt.update(nrn, g); mx.eval(nrn, opt.state)
    m.blocks[f.cb].fc2 = F2[f.cb]; return {"u": np.array(nrn["u"], np.float64), "b0": float(nrn["b0"].item()), "w": np.array(nrn["w"], np.float64)}
