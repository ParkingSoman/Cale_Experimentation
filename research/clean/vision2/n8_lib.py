"""Compounding: many families collapsed at once, parents solved in the real (edited) model weights."""
import json, os, sys, time, glob
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from convnext import *
t0 = time.time(); m = load_convnext(); X, y = load_images(); ST = m.stage_of; NB = 18
H = [m.blocks[b].fc1.weight.shape[0] for b in range(NB)]; GM = [np.array(m.blocks[b].gamma, np.float64) for b in range(NB)]
gelu = nn.gelu; rng = np.random.default_rng(8); CD = os.path.join(HF, "imagenet", "n7cache")
ORIG = [(np.array(m.blocks[b].fc1.weight), np.array(m.blocks[b].fc1.bias), np.array(m.blocks[b].fc2.weight), np.array(m.blocks[b].fc2.bias)) for b in range(NB)]
def lin(W, b): f = nn.Linear(W.shape[1], W.shape[0]); f.weight = mx.array(W.astype(np.float32)); f.bias = mx.array(b.astype(np.float32)); return f
def restore():
    for b in range(NB): m.blocks[b].fc1 = lin(ORIG[b][0], ORIG[b][1]); m.blocks[b].fc2 = lin(ORIG[b][2], ORIG[b][3])
def set_neuron(b, n, u, b0, w):
    W1 = np.array(m.blocks[b].fc1.weight); B1 = np.array(m.blocks[b].fc1.bias); W2 = np.array(m.blocks[b].fc2.weight); B2 = np.array(m.blocks[b].fc2.bias)
    W1[n] = u; B1[n] = b0; W2[:, n] = w; m.blocks[b].fc1 = lin(W1, B1); m.blocks[b].fc2 = lin(W2, B2)
def cut(b, n, mu):
    W2 = np.array(m.blocks[b].fc2.weight); B2 = np.array(m.blocks[b].fc2.bias); B2 = B2 + W2[:, n] * mu; W2[:, n] = 0; m.blocks[b].fc2 = lin(W2, B2)
def batches(n, k=50): return [(i, min(i + k, n)) for i in range(0, n, k)]
def gelu_d(z): return 0.5 * (1 + mx.erf(z / 2 ** .5)) + z * mx.exp(-z * z / 2) / (2 * np.pi) ** .5
def to_block(xs, b):
    x = m.stem_norm(m.stem(xs))
    for i in range(b):
        if i > 0 and ST[i] != ST[i - 1]: x = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](x))
        x = m.blocks[i](x)
    if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
    return x
def logits(I): return mx.concatenate([m(mx.array(norm_img(np.asarray(X[I[a:b]])))) for a, b in batches(len(I))])
def klv(lg, T):
    lp = lg - mx.logsumexp(lg, axis=1, keepdims=True); tp = T - mx.logsumexp(T, axis=1, keepdims=True); return np.array((mx.exp(tp) * (tp - lp)).sum(1))

class Fam:
    def __init__(self, key):
        z = np.load(os.path.join(CD, key + ".npz")); self.key = key
        self.p = tuple(int(v) for v in z["p"]); self.cs = [tuple(int(v) for v in c) for c in z["cs"]]; self.pb, self.pn = self.p; self.cb = self.pb + 1
        self.tr, self.ev, self.TT, self.mu = z["tr"], z["ev"], mx.array(z["TT"]), z["mu"]; self.L = {"u": z["Lu"], "b0": float(z["Lb"]), "w": z["Lw"]}
        idx = np.arange(len(self.tr)); self.gix, self.vix = mx.array(idx[idx % 3 != 0]), mx.array(idx[idx % 3 == 0])
    def current(self):
        W1 = np.array(m.blocks[self.pb].fc1.weight); B1 = np.array(m.blocks[self.pb].fc1.bias); W2 = np.array(m.blocks[self.pb].fc2.weight)
        return {"u": W1[self.pn].astype(np.float64), "b0": float(B1[self.pn]), "w": W2[:, self.pn].astype(np.float64)}
    def prepare(self):
        self.mask = mx.array((np.arange(H[self.pb]) != self.pn).astype(np.float32))
        self.XT = mx.concatenate([to_block(mx.array(norm_img(np.asarray(X[self.tr[a:b]]))), self.pb) for a, b in batches(len(self.tr))]); mx.eval(self.XT)
    def block_pb(self, x, u, b0, w):
        blk = m.blocks[self.pb]; pre = blk.pre(x); hh = gelu(blk.fc1(pre)) * self.mask; z = pre @ u + b0; h = gelu(z)
        return x + blk.gamma * (blk.fc2(hh) + h[..., None] * w), pre, z, h
    def suffix(self, h):
        for i in range(self.cb, NB):
            if ST[i] != ST[i - 1]: h = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](h))
            h = m.blocks[i](h)
        return m.head(h)
    def trainkl(self, th, ix=None):
        ix = self.vix if ix is None else ix; X_, T_ = self.XT[ix], self.TT[ix]; out = []
        u = mx.array(np.asarray(th["u"], np.float32)); w = mx.array(np.asarray(th["w"], np.float32))
        for a, b in batches(X_.shape[0]): s1, _, _, _ = self.block_pb(X_[a:b], u, th["b0"], w); out.append(self.suffix(s1))
        return float(klv(mx.concatenate(out), T_).mean())
    def gn_stats(self, th, S=4):
        gam = mx.array(GM[self.pb].astype(np.float32)); u = mx.array(np.asarray(th["u"], np.float32)); w = mx.array(np.asarray(th["w"], np.float32)); A = None; bv = None
        X_, T_ = self.XT[self.gix], self.TT[self.gix]
        for a, b in batches(X_.shape[0], 25):
            s1, pre, z, h = self.block_pb(X_[a:b], u, th["b0"], w); lg = self.suffix(s1); p = mx.softmax(lg, axis=1); ps = mx.softmax(T_[a:b], axis=1)
            pn_ = np.array(p, np.float64); pn_ /= pn_.sum(1, keepdims=True); n_ = b - a; dz = gelu_d(z)
            for k, v in enumerate([p - ps] + [mx.eye(1000)[mx.array([rng.choice(1000, p=q) for q in pn_])] - p for _ in range(S)]):
                _, (g1,) = mx.vjp(self.suffix, [s1], [v]); gwh = g1 @ (gam * w)
                phi = np.array(mx.concatenate([mx.einsum("np,npk->nk", (gwh * dz).reshape(n_, -1), pre.reshape(n_, -1, pre.shape[-1])), (gwh * dz).reshape(n_, -1).sum(1, keepdims=True),
                                               (h.reshape(n_, -1, 1) * g1.reshape(n_, -1, g1.shape[-1])).sum(1) * gam], axis=1), np.float64)
                if A is None: A = np.zeros((phi.shape[1],) * 2); bv = np.zeros(phi.shape[1])
                if k == 0: bv += phi.sum(0)
                else: A += phi.T @ phi / S
        return A, bv
    def gn(self, th, nit=3, lms=(1e-1, 1, 10, 100), als=(.5, 1.0)):
        C = len(th["u"]); kv0 = self.trainkl(th)
        for it in range(nit):
            A, bv = self.gn_stats(th); dg = np.diag(A); best = (kv0, None)
            for lm in lms:
                d = -np.linalg.solve(A + lm * np.diag(dg + 1e-9 * dg.max()), bv)
                for al in als:
                    c = {"u": th["u"] + al * d[:C], "b0": th["b0"] + al * d[C], "w": th["w"] + al * d[C + 1:]}; kv = self.trainkl(c)
                    if kv < best[0]: best = (kv, c)
            if best[1] is None: break
            th, kv0 = best[1], best[0]
        return th
    def train(self, th, epochs=12, lr=3e-4):
        P = {k: mx.array(np.asarray(v, np.float32)) for k, v in th.items()}
        def loss(P, xb, tb):
            s1, _, _, _ = self.block_pb(xb, P["u"], P["b0"], P["w"]); lg = self.suffix(s1)
            lp = lg - mx.logsumexp(lg, axis=1, keepdims=True); tp = tb - mx.logsumexp(tb, axis=1, keepdims=True); return (mx.exp(tp) * (tp - lp)).sum(1).mean()
        lvg = mx.value_and_grad(loss); opt = optim.Adam(learning_rate=lr); gi = np.array(self.gix)
        for ep in range(epochs):
            perm = rng.permutation(gi)
            for a in range(0, len(perm), 50):
                j = mx.array(np.sort(perm[a:a + 50])); l, g = lvg(P, self.XT[j], self.TT[j]); opt.update(P, g); mx.eval(P, opt.state)
        return {"u": np.array(P["u"], np.float64), "b0": float(P["b0"].item()), "w": np.array(P["w"], np.float64)}
