"""N10: whole block 16 at 50% removed; block-15 parents solved by bank-GN (all at once) vs trained. VISION-LOG N10.
usage: python n10_block.py NPARENTS(0=all) NIT TRAIN_EPOCHS"""
import json, os, sys, time
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from convnext import *
NPAR, NIT, TEP = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
t0 = time.time(); m = load_convnext(); X, y = load_images(); ST = m.stage_of; NB = 18; gelu = nn.gelu; rng = np.random.default_rng(10)
H = [m.blocks[b].fc1.weight.shape[0] for b in range(NB)]
rank = np.zeros(len(y), int)
for c in range(1000): ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
CAL = np.where(rank < 1)[0]; EV = np.where((rank >= 20) & (rank < 23))[0]; TR = np.sort(rng.choice(np.where(rank < 20)[0], 1500, replace=False))
MAP = json.load(open(os.path.join(ROOT, "research/clean/map/network_map_v2.json")))["neurons"]
def lin(W, b): f = nn.Linear(W.shape[1], W.shape[0]); f.weight = mx.array(np.asarray(W, np.float32)); f.bias = mx.array(np.asarray(b, np.float32)); return f
def batches(n, k=50): return [(i, min(i + k, n)) for i in range(0, n, k)]
def gelu_d(z): return 0.5 * (1 + mx.erf(z / 2 ** .5)) + z * mx.exp(-z * z / 2) / (2 * np.pi) ** .5
def to15(xs):
    x = m.stem_norm(m.stem(xs))
    for i in range(15):
        if i > 0 and ST[i] != ST[i - 1]: x = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](x))
        x = m.blocks[i](x)
    return m.down[2](m.down_norm[2](x))
def suffix(h): return m.head(m.blocks[17](m.blocks[16](h)))
def klv(lg, T):
    lp = lg - mx.logsumexp(lg, axis=1, keepdims=True); tp = T - mx.logsumexp(T, axis=1, keepdims=True); return np.array((mx.exp(tp) * (tp - lp)).sum(1))
B15 = m.blocks[15]; W1o, b1o, W2o, b2o = [np.array(a, np.float64) for a in (B15.fc1.weight, B15.fc1.bias, B15.fc2.weight, B15.fc2.bias)]; gam = np.array(B15.gamma, np.float64)
F16 = m.blocks[16].fc2; W216, b216 = np.array(F16.weight), np.array(F16.bias)
# energy of block-16 neurons and their means (CAL)
X15c = mx.concatenate([to15(mx.array(norm_img(np.asarray(X[CAL[a:b]])))) for a, b in batches(len(CAL))])
h16 = np.concatenate([np.array(gelu(m.blocks[16].fc1(m.blocks[16].pre(B15(X15c[a:b]))))).reshape(-1, H[16]) for a, b in batches(len(CAL))])
mu16 = h16.mean(0); en = np.linalg.norm(h16, axis=0) * np.linalg.norm(W216 * np.array(m.blocks[16].gamma)[:, None], axis=0); del h16
REM = np.argsort(en)[:H[16] // 2]; par = {}
for c in REM:
    p = tuple(MAP[f"16_{c}"]["parent"])
    if p[0] == 15: par.setdefault(p[1], []).append(int(c))
J = sorted(par)
if NPAR: J = sorted(rng.choice(J, NPAR, replace=False).tolist())
print("removed", len(REM), "children with block-15 parent", sum(len(v) for v in par.values()), "parents solved", len(J), f"{time.time() - t0:.0f}s", flush=True)
W2c = W216.copy(); b2c = b216 + W216[:, REM] @ mu16[REM]; W2c[:, REM] = 0; F16cut = lin(W2c, b2c)
def set15(U, bJ, WJ):   # U (k,C) model units, WJ (C,k) model units (pre-gamma)
    W1 = W1o.copy(); b1 = b1o.copy(); W2 = W2o.copy(); W1[J] = U; b1[J] = bJ; W2[:, J] = WJ; B15.fc1 = lin(W1, b1); B15.fc2 = lin(W2, b2o)
def logits_full(I): return mx.concatenate([m(mx.array(norm_img(np.asarray(X[I[a:b]])))) for a, b in batches(len(I))])
# intact teachers
m.blocks[16].fc2 = F16; LEV = logits_full(EV); XT = mx.concatenate([to15(mx.array(norm_img(np.asarray(X[TR[a:b]])))) for a, b in batches(len(TR))]); TT = mx.concatenate([suffix(B15(XT[a:b])) for a, b in batches(len(TR))]); mx.eval(LEV, XT, TT)
m.blocks[16].fc2 = F16cut; gix = np.arange(len(TR))[np.arange(len(TR)) % 3 != 0]; vix = np.arange(len(TR))[np.arange(len(TR)) % 3 == 0]
def valkl(): return float(np.mean(np.concatenate([klv(suffix(B15(XT[mx.array(vix[a:b])])), TT[mx.array(vix[a:b])]) for a, b in batches(len(vix))])))
def evalshare(): return float(klv(logits_full(EV), LEV).mean())
U0, bJ0, WJ0 = W1o[J].copy(), b1o[J].copy(), W2o[:, J].copy()
set15(U0, bJ0, WJ0); KCUT = evalshare(); print("eval KL cut %.4f" % KCUT, f"{time.time() - t0:.0f}s", flush=True)
res = {"n_parents": len(J), "kl_cut": KCUT}
# ---------------- joint bank-GN, matrix-free conjugate gradients over ALL parents at once
U, bJ, WJ = U0.copy(), bJ0.copy(), WJ0.copy(); S = int(os.environ.get('TOPK', 6)); Jm = mx.array(J); gm = mx.array(gam.astype(np.float32)); C = 768; k = len(J); gi = mx.array(gix)
def bank(U, bJ, WJ):
    set15(U, bJ, WJ); Gs = [[] for _ in range(S + 1)]; A15, Hs, Dz = [], [], []
    for a, b in batches(len(gix), 50):
        x = XT[gi[a:b]]; pre = B15.pre(x); z = B15.fc1(pre)[..., Jm]; s1 = B15(x); lg = suffix(s1); p = mx.softmax(lg, axis=1); ps = mx.softmax(TT[gi[a:b]], axis=1)
        pn_ = np.array(p, np.float64); pn_ /= pn_.sum(1, keepdims=True); n_ = b - a
        top = np.argsort(-pn_, axis=1)[:, :S]; vlist = [p - ps]
        for kk in range(S):                                   # exact Fisher on each image's top-S classes: v = sqrt(p_k) (e_k - p)
            ek = mx.eye(1000)[mx.array(top[:, kk])]; vlist.append(mx.sqrt(mx.array(pn_[np.arange(n_), top[:, kk]].astype(np.float32)))[:, None] * (ek - p))
        for s_, v in enumerate(vlist):
            _, (g1,) = mx.vjp(suffix, [s1], [v]); Gs[s_].append((g1 * gm).reshape(n_, -1, C))
        A15.append(pre.reshape(n_, -1, C)); Hs.append(gelu(z).reshape(n_, -1, k)); Dz.append(gelu_d(z).reshape(n_, -1, k))
    Gs = [mx.concatenate(g) for g in Gs]; A15, Hs, Dz = mx.concatenate(A15), mx.concatenate(Hs), mx.concatenate(Dz); mx.eval(Gs, A15, Hs, Dz)
    Wg = mx.array(WJ.astype(np.float32)); Es = [(g @ Wg) * Dz for g in Gs]; mx.eval(Es); return Gs, Es, A15, Hs
def Jd(g, e, A15, Hs, dU, db, dW): return (Hs * (g @ dW) + e * (A15 @ dU.T + db)).sum((1, 2))                      # (n,)
def JTr(g, e, A15, Hs, r):
    rH = Hs * r[:, None, None]; rE = e * r[:, None, None]
    return mx.einsum("npk,npc->kc", rE, A15), rE.sum((0, 1)), mx.einsum("npk,npc->ck", rH, g)                     # dU (k,C), db (k,), dW (C,k)
def solve(Gs, Es, A15, Hs, lm, iters=25):
    gU, gb, gW = JTr(Gs[0], Es[0], A15, Hs, mx.ones((Hs.shape[0],)))
    dg = [mx.zeros((k, C)), mx.zeros((k,)), mx.zeros((C, k))]                                                       # diagonal of A (for damping)
    for s_ in range(1, S + 1):
        dg[0] = dg[0] + mx.einsum("npk,npc->nkc", Es[s_], A15).square().sum(0) / S if False else dg[0]
    def Av(v):
        out = [mx.zeros_like(t) for t in v]
        for s_ in range(1, S + 1):
            r = Jd(Gs[s_], Es[s_], A15, Hs, *v); t = JTr(Gs[s_], Es[s_], A15, Hs, r); out = [o + q for o, q in zip(out, t)]
        return out
    # damping scale: per-group mean curvature from a random probe
    pr = [mx.array(rng.standard_normal(t.shape).astype(np.float32)) for t in (gU, gb, gW)]; Ap = Av(pr)
    sc = [float((a * p_).sum() / (p_ * p_).sum()) for a, p_ in zip(Ap, pr)]
    def M(v): q = Av(v); return [qq + lm * s * vv for qq, s, vv in zip(q, sc, v)]
    b_ = [-gU, -gb, -gW]; x = [mx.zeros_like(t) for t in b_]; r = [t for t in b_]; p_ = [t for t in r]; rr = sum(float((t * t).sum()) for t in r)
    for _ in range(iters):
        Mp = M(p_); al = rr / max(sum(float((a * b).sum()) for a, b in zip(p_, Mp)), 1e-30)
        x = [xx + al * pp for xx, pp in zip(x, p_)]; r = [rr_ - al * mp for rr_, mp in zip(r, Mp)]; rn = sum(float((t * t).sum()) for t in r)
        p_ = [rr_ + (rn / rr) * pp for rr_, pp in zip(r, p_)]; rr = rn; mx.eval(x, r, p_)
    return [np.array(t, np.float64) for t in x]
if os.environ.get("DEBUG"):
    def trkl():
        return float(np.mean(np.concatenate([klv(suffix(B15(XT[gi[a:b]])), TT[gi[a:b]]) for a, b in batches(len(gix))])))
    set15(U, bJ, WJ); k0 = trkl(); Gs, Es, A15, Hs = bank(U, bJ, WJ); gU, gb, gW = [np.array(t, np.float64) for t in JTr(Gs[0], Es[0], A15, Hs, mx.ones((Hs.shape[0],)))]
    gn2 = (gU ** 2).sum() + (gb ** 2).sum() + (gW ** 2).sum(); print("train KL", k0, "|g|^2", gn2, flush=True)
    for eps in (1e-1, 1, 10, 100):
        set15(U - eps * gU, bJ - eps * gb, WJ - eps * gW); k1 = trkl(); print(" step -eps*g eps", eps, "trainKL change %.3e  predicted (mean over n) %.3e" % (k1 - k0, -eps * gn2 / len(gix)), flush=True)
    sys.exit()
for it in range(NIT):
    set15(U, bJ, WJ); kv0 = valkl(); Gs, Es, A15, Hs = bank(U, bJ, WJ); best = (kv0, None)
    for lm in (100, 1e3):
        dU, db, dW = solve(Gs, Es, A15, Hs, lm)
        for al in (.3, .6):
            set15(U + al * dU, bJ + al * db, WJ + al * dW); kv = valkl(); print("   lm", lm, "al", al, "val %.5f (base %.5f) |dW| %.2f" % (kv, kv0, np.linalg.norm(al * dW)), flush=True)
            if kv < best[0]: best = (kv, (lm, al, dU, db, dW))
    del Gs, Es, A15, Hs
    if best[1] is None: print("joint GN: no improving step at it", it, flush=True); break
    lm, al, dU, db, dW = best[1]; U, bJ, WJ = U + al * dU, bJ + al * db, WJ + al * dW
    set15(U, bJ, WJ); res[f"jointgn{it + 1}"] = (KCUT - evalshare()) / KCUT; print("joint GN it", it + 1, "lm al", lm, al, "val %.5f -> %.5f" % (kv0, best[0]), "eval share %.3f" % res[f"jointgn{it + 1}"], f"{time.time() - t0:.0f}s", flush=True)
# ---------------- trained reference: Adam on the same parameters, same images
P = {"U": mx.array(U0.astype(np.float32)), "b": mx.array(bJ0.astype(np.float32)), "W": mx.array(WJ0.astype(np.float32))}; set15(U0, bJ0, WJ0)
mask = mx.array((~np.isin(np.arange(H[15]), J)).astype(np.float32))
def loss(P, xb, tb):
    pre = B15.pre(xb); hall = gelu(B15.fc1(pre)) * mask; hj = gelu(pre @ P["U"].T + P["b"]); s1 = xb + B15.gamma * (B15.fc2(hall) + hj @ P["W"].T)
    lg = suffix(s1); lp = lg - mx.logsumexp(lg, axis=1, keepdims=True); tp = tb - mx.logsumexp(tb, axis=1, keepdims=True); return (mx.exp(tp) * (tp - lp)).sum(1).mean()
lvg = mx.value_and_grad(loss); opt = optim.Adam(learning_rate=float(os.environ.get("LR", 3e-4)))
for ep in range(TEP):
    perm = rng.permutation(gix)
    for a in range(0, len(perm), 50):
        j = mx.array(np.sort(perm[a:a + 50])); l, g = lvg(P, XT[j], TT[j]); opt.update(P, g); mx.eval(P, opt.state)
    if ep + 1 in (4, 8, 16, 30, 50, TEP):
        set15(np.array(P["U"], np.float64), np.array(P["b"], np.float64), np.array(P["W"], np.float64)); res[f"trained{ep + 1}"] = (KCUT - evalshare()) / KCUT
        print("trained", ep + 1, "eval share %.3f" % res[f"trained{ep + 1}"], f"{time.time() - t0:.0f}s", flush=True)
json.dump(res, open(os.path.join(RES, f"vision-n10-block-{NPAR}{os.environ.get("TAG", "")}.json"), "w"), indent=1); print(res)
