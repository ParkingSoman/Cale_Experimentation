"""N11: compounding across blocks. Child blocks (deep -> shallow), 50% removed each; each block's kept map parents in
the block before are solved jointly (joint GN, top-K Fisher, matrix-free CG) or trained (Adam), sequentially.
usage: python n11_multi.py MODE(gn|tr) CHILD_BLOCKS(e.g. 17,16) NIT_or_EPOCHS"""
import json, os, sys, time
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from convnext import *
MODE, CBS, NIT = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], int(sys.argv[3])
t0 = time.time(); m = load_convnext(); X, y = load_images(); ST = m.stage_of; NB = 18; gelu = nn.gelu; rng = np.random.default_rng(int(os.environ.get("SEED", 11)))
H = [m.blocks[b].fc1.weight.shape[0] for b in range(NB)]
rank = np.zeros(len(y), int)
for c in range(1000): ix = np.where(y == c)[0]; rank[ix] = np.arange(len(ix))
CAL = np.where(rank < 1)[0]; EV = np.where((rank >= 20) & (rank < 23))[0]; TR = np.sort(rng.choice(np.where(rank < 20)[0], 1200, replace=False))
MAP = json.load(open(os.path.join(ROOT, "research/clean/map/network_map_v2.json")))["neurons"]
def lin(W, b): f = nn.Linear(W.shape[1], W.shape[0]); f.weight = mx.array(np.asarray(W, np.float32)); f.bias = mx.array(np.asarray(b, np.float32)); return f
def batches(n, k=50): return [(i, min(i + k, n)) for i in range(0, n, k)]
def gelu_d(z): return 0.5 * (1 + mx.erf(z / 2 ** .5)) + z * mx.exp(-z * z / 2) / (2 * np.pi) ** .5
def to_block(xs, b):
    x = m.stem_norm(m.stem(xs))
    for i in range(b):
        if i > 0 and ST[i] != ST[i - 1]: x = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](x))
        x = m.blocks[i](x)
    if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
    return x
def suffix_from(pb):
    def f(h):
        for i in range(pb + 1, NB):
            if ST[i] != ST[i - 1]: h = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](h))
            h = m.blocks[i](h)
        return m.head(h)
    return f
def klv(lg, T):
    lp = lg - mx.logsumexp(lg, axis=1, keepdims=True); tp = T - mx.logsumexp(T, axis=1, keepdims=True); return np.array((mx.exp(tp) * (tp - lp)).sum(1))
def logits_full(I): return mx.concatenate([m(mx.array(norm_img(np.asarray(X[I[a:b]])))) for a, b in batches(len(I))])
LEV = logits_full(EV); LTR = logits_full(TR); mx.eval(LEV, LTR)
# energies and means (CAL) for every child block; removal = lowest-energy half
REM, MU = {}, {}
x = None
for cb in CBS:
    hs = []
    for a, b in batches(len(CAL)):
        xb = to_block(mx.array(norm_img(np.asarray(X[CAL[a:b]]))), cb); hs.append(np.array(gelu(m.blocks[cb].fc1(m.blocks[cb].pre(xb)))).reshape(-1, H[cb]))
    hs = np.concatenate(hs); MU[cb] = hs.mean(0); en = np.linalg.norm(hs, axis=0) * np.linalg.norm(np.array(m.blocks[cb].fc2.weight) * np.array(m.blocks[cb].gamma)[:, None], axis=0)
    REM[cb] = np.argsort(en)[:H[cb] // 2]
for cb in CBS:
    W2 = np.array(m.blocks[cb].fc2.weight); b2 = np.array(m.blocks[cb].fc2.bias); b2 = b2 + W2[:, REM[cb]] @ MU[cb][REM[cb]]; W2[:, REM[cb]] = 0; m.blocks[cb].fc2 = lin(W2, b2)
KCUT = float(klv(logits_full(EV), LEV).mean()); res = {"blocks": CBS, "kl_cut": KCUT, "mode": MODE}; print("cut eval KL %.4f" % KCUT, f"{time.time() - t0:.0f}s", flush=True)
gix = np.arange(len(TR))[np.arange(len(TR)) % 3 != 0]; vix = np.arange(len(TR))[np.arange(len(TR)) % 3 == 0]; TOPK = int(os.environ.get('TOPK', 4))
for cb in CBS:                                       # deep -> shallow
    pb = cb - 1; BP = m.blocks[pb]; rem_pb = set(REM.get(pb, np.array([], int)).tolist())
    J = sorted({MAP[f"{cb}_{c}"]["parent"][1] for c in REM[cb] if MAP[f"{cb}_{c}"]["parent"][0] == pb} - rem_pb); k = len(J); Jm = mx.array(J)
    C = BP.fc1.weight.shape[1]; gam = np.array(BP.gamma, np.float64); gm = mx.array(gam.astype(np.float32)); suffix = suffix_from(pb)
    W1o, b1o, W2o, b2o = [np.array(a, np.float64) for a in (BP.fc1.weight, BP.fc1.bias, BP.fc2.weight, BP.fc2.bias)]
    XT = mx.concatenate([to_block(mx.array(norm_img(np.asarray(X[TR[a:b]]))), pb) for a, b in batches(len(TR))]); TT = LTR; mx.eval(XT)
    def setp(U, bJ, WJ):
        W1 = W1o.copy(); b1 = b1o.copy(); W2 = W2o.copy(); W1[J] = U; b1[J] = bJ; W2[:, J] = WJ; BP.fc1 = lin(W1, b1); BP.fc2 = lin(W2, b2o)
    def valkl(): return float(np.mean(np.concatenate([klv(suffix(BP(XT[mx.array(vix[a:b])])), TT[mx.array(vix[a:b])]) for a, b in batches(len(vix))])))
    U, bJ, WJ = W1o[J].copy(), b1o[J].copy(), W2o[:, J].copy(); gi = mx.array(gix)
    print(f" block {pb}: {k} parents for block {cb}'s removed children", f"{time.time() - t0:.0f}s", flush=True)
    if MODE == "gn":
        for it in range(NIT):
            setp(U, bJ, WJ); kv0 = valkl(); Gs = [[] for _ in range(TOPK + 1)]; A15, Hs, Dz = [], [], []
            for a, b in batches(len(gix), 50):
                xb = XT[gi[a:b]]; pre = BP.pre(xb); z = BP.fc1(pre)[..., Jm]; s1 = BP(xb); lg = suffix(s1); p = mx.softmax(lg, axis=1); ps = mx.softmax(TT[gi[a:b]], axis=1)
                pn_ = np.array(p, np.float64); pn_ /= pn_.sum(1, keepdims=True); n_ = b - a; top = np.argsort(-pn_, axis=1)[:, :TOPK]; vl = [p - ps]
                for kk in range(TOPK): vl.append(mx.sqrt(mx.array(pn_[np.arange(n_), top[:, kk]].astype(np.float32)))[:, None] * (mx.eye(1000)[mx.array(top[:, kk])] - p))
                for s_, v in enumerate(vl): _, (g1,) = mx.vjp(suffix, [s1], [v]); Gs[s_].append((g1 * gm).reshape(n_, -1, C))
                A15.append(pre.reshape(n_, -1, C)); Hs.append(gelu(z).reshape(n_, -1, k)); Dz.append(gelu_d(z).reshape(n_, -1, k))
            Gs = [mx.concatenate(g) for g in Gs]; A15, Hs, Dz = mx.concatenate(A15), mx.concatenate(Hs), mx.concatenate(Dz); Wg = mx.array(WJ.astype(np.float32)); Es = [(g @ Wg) * Dz for g in Gs]; mx.eval(Gs, Es, A15, Hs)
            Jd = lambda g, e, dU, db, dW: (Hs * (g @ dW) + e * (A15 @ dU.T + db)).sum((1, 2))
            def JTr(g, e, r): rH = Hs * r[:, None, None]; rE = e * r[:, None, None]; return mx.einsum("npk,npc->kc", rE, A15), rE.sum((0, 1)), mx.einsum("npk,npc->ck", rH, g)
            def Av(v):
                out = [mx.zeros_like(t) for t in v]
                for s_ in range(1, TOPK + 1): t = JTr(Gs[s_], Es[s_], Jd(Gs[s_], Es[s_], *v)); out = [o + q for o, q in zip(out, t)]
                return out
            gU, gb, gW = JTr(Gs[0], Es[0], mx.ones((Hs.shape[0],))); pr = [mx.array(rng.standard_normal(t.shape).astype(np.float32)) for t in (gU, gb, gW)]
            sc = [float((a_ * p_).sum() / (p_ * p_).sum()) for a_, p_ in zip(Av(pr), pr)]; best = (kv0, None)
            if os.environ.get("DIAG"):            # exact diag(A) per weight, in chunks of parents
                dU_, db_, dW_ = mx.zeros((k, C)), mx.zeros((k,)), mx.zeros((C, k))
                for s_ in range(1, TOPK + 1):
                    for c0 in range(0, k, 32):
                        sl = slice(c0, min(c0 + 32, k)); hh = Hs[..., sl]; ee = Es[s_][..., sl]
                        dW_[:, sl] = dW_[:, sl] + mx.einsum("npk,npc->nck", hh, Gs[s_]).square().sum(0)
                        dU_[sl] = dU_[sl] + mx.einsum("npk,npc->nkc", ee, A15).square().sum(0); db_[sl] = db_[sl] + ee.sum(1).square().sum(0)
                    mx.eval(dU_, db_, dW_)
                dgs = [dU_ + 1e-8 * mx.max(dU_), db_ + 1e-8 * mx.max(db_), dW_ + 1e-8 * mx.max(dW_)]
            for lm in [float(v) for v in os.environ.get('LMS', '100,1000').split(',')]:
                M = (lambda v: [q + lm * d_ * vv for q, d_, vv in zip(Av(v), dgs, v)]) if os.environ.get('DIAG') else (lambda v: [q + lm * s * vv for q, s, vv in zip(Av(v), sc, v)])
                bb = [-gU, -gb, -gW]; xx = [mx.zeros_like(t) for t in bb]; r = list(bb); pp = list(r); rr = sum(float((t * t).sum()) for t in r)
                for _ in range(25):
                    Mp = M(pp); al_ = rr / max(sum(float((a_ * b_).sum()) for a_, b_ in zip(pp, Mp)), 1e-30)
                    xx = [q + al_ * w for q, w in zip(xx, pp)]; r = [q - al_ * w for q, w in zip(r, Mp)]; rn = sum(float((t * t).sum()) for t in r); pp = [q + (rn / rr) * w for q, w in zip(r, pp)]; rr = rn; mx.eval(xx, r, pp)
                dU, db, dW = [np.array(t, np.float64) for t in xx]
                for al in [float(v) for v in os.environ.get('ALS', '.3,.6').split(',')]:
                    setp(U + al * dU, bJ + al * db, WJ + al * dW); kv = valkl()
                    if kv < best[0]: best = (kv, (al * dU, al * db, al * dW))
            del Gs, Es, A15, Hs, Dz
            if best[1] is None: print(f"   it {it + 1}: no improving step (val {kv0:.5f})", flush=True); break
            U, bJ, WJ = U + best[1][0], bJ + best[1][1], WJ + best[1][2]; print(f"   it {it + 1} val {kv0:.5f} -> {best[0]:.5f}", f"{time.time() - t0:.0f}s", flush=True)
    else:
        P = {"U": mx.array(U.astype(np.float32)), "b": mx.array(bJ.astype(np.float32)), "W": mx.array(WJ.astype(np.float32))}; setp(U, bJ, WJ)
        mask = mx.array((~np.isin(np.arange(H[pb]), J)).astype(np.float32))
        def loss(P, xb, tb):
            pre = BP.pre(xb); hall = gelu(BP.fc1(pre)) * mask; s1 = xb + BP.gamma * (BP.fc2(hall) + gelu(pre @ P["U"].T + P["b"]) @ P["W"].T)
            lg = suffix(s1); lp = lg - mx.logsumexp(lg, axis=1, keepdims=True); tp = tb - mx.logsumexp(tb, axis=1, keepdims=True); return (mx.exp(tp) * (tp - lp)).sum(1).mean()
        lvg = mx.value_and_grad(loss); opt = optim.Adam(learning_rate=3e-4)
        def vk(P):
            out = []
            for a, b in batches(len(vix)):
                xb = XT[mx.array(vix[a:b])]; pre = BP.pre(xb); hall = gelu(BP.fc1(pre)) * mask; s1 = xb + BP.gamma * (BP.fc2(hall) + gelu(pre @ P["U"].T + P["b"]) @ P["W"].T); out.append(klv(suffix(s1), TT[mx.array(vix[a:b])]))
            return float(np.mean(np.concatenate(out)))
        bestP = (vk(P), {k: np.array(v) for k, v in P.items()}, 0)
        for ep in range(NIT):
            perm = rng.permutation(gix)
            for a in range(0, len(perm), 50): j = mx.array(np.sort(perm[a:a + 50])); l, g = lvg(P, XT[j], TT[j]); opt.update(P, g); mx.eval(P, opt.state)
            v_ = vk(P)
            if v_ < bestP[0]: bestP = (v_, {k: np.array(v) for k, v in P.items()}, ep + 1)
        print(f"   early stop: best epoch {bestP[2]} of {NIT}", flush=True)
        U, bJ, WJ = bestP[1]["U"].astype(np.float64), bestP[1]["b"].astype(np.float64), bestP[1]["W"].astype(np.float64)
    setp(U, bJ, WJ); del XT
    res[f"after_block{pb}"] = (KCUT - float(klv(logits_full(EV), LEV).mean())) / KCUT; print(f" block {pb} done: eval share {res[f'after_block{pb}']:.3f}", f"{time.time() - t0:.0f}s", flush=True)
    if os.environ.get("STOP1"): sys.exit()
np.savez(os.path.join(HF, "imagenet", f"n11_{MODE}_{'_'.join(map(str, CBS))}.npz"), **{f"{t}{b}": np.array(getattr(m.blocks[b], n).__dict__.get('_x', getattr(getattr(m.blocks[b], n), a)), np.float32) for b in range(min(CBS) - 1, NB) for t, n, a in (("W1_", "fc1", "weight"), ("b1_", "fc1", "bias"), ("W2_", "fc2", "weight"), ("b2_", "fc2", "bias"))})
json.dump(res, open(os.path.join(RES, f"vision-n11-{MODE}-{'_'.join(map(str, CBS))}-{NIT}.json"), "w"), indent=1); print(res)
