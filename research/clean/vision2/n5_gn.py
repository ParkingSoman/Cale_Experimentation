"""N5: Gauss-Newton on the full neuron (u, b0, w) vs training. Rules: VISION-LOG.md N5."""
import sys
from n_harness import *
rng = np.random.default_rng(5); S = 4; NIT = 3
def gelu_d(z): return 0.5 * (1 + mx.erf(z / 2 ** .5)) + z * mx.exp(-z * z / 2) / (2 * np.pi) ** .5
def suffix(f, h):
    for i in range(f.cb, NB):
        if ST[i] != ST[i - 1]: h = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](h))
        h = m.blocks[i](h)
    return m.head(h)
def gn_step(f, th, X_, T_):
    """returns gradient b and curvature A (Fisher-GN) for theta = [u (C), b0, w (C)] in model units."""
    C = len(th["u"]); D = 2 * C + 1; A = np.zeros((D, D)); bvec = np.zeros(D); gam = mx.array(GM[f.pb].astype(np.float32))
    m.blocks[f.cb].fc2 = f.fc2cut; blk = m.blocks[f.pb]; u = mx.array(th["u"]); w = mx.array(th["w"])
    for a, b in batches(X_.shape[0], 25):
        x = X_[a:b]; pre = blk.pre(x); hall = gelu(blk.fc1(pre)); hall = hall - hall[..., f.pn:f.pn + 1] * mx.array(np.eye(H[f.pb])[f.pn].astype(np.float32))
        z = pre @ u + th["b0"]; h = gelu(z); s1 = x + blk.gamma * (blk.fc2(hall) + h[..., None] * w)
        lg = suffix(f, s1); p = mx.softmax(lg, axis=1); ps = mx.softmax(T_[a:b], axis=1)
        vs = [p - ps]; pn_ = np.array(p, np.float64); pn_ /= pn_.sum(1, keepdims=True)
        for s in range(S): vs.append(mx.eye(1000)[mx.array([rng.choice(1000, p=q) for q in pn_])] - p)
        gw = (gam * w); dz = gelu_d(z)
        for k, v in enumerate(vs):
            _, (g1,) = mx.vjp(lambda hh: suffix(f, hh), [s1], [v])
            gwh = (g1 @ gw)                                                       # (n,R,R): g1 . (gamma*w)
            phi_u = mx.einsum("nrc,nrk->nk", (gwh * dz).reshape(b - a, -1, 1), pre.reshape(b - a, -1, pre.shape[-1])).reshape(b - a, -1)
            phi_b = (gwh * dz).reshape(b - a, -1).sum(1, keepdims=True)
            phi_w = (h.reshape(b - a, -1, 1) * g1.reshape(b - a, -1, g1.shape[-1])).sum(1) * gam
            phi = np.array(mx.concatenate([phi_u, phi_b, phi_w], axis=1), np.float64)
            if k == 0: bvec += phi.sum(0)
            else: A += phi.T @ phi / S
    m.blocks[f.cb].fc2 = F2[f.cb]; return A, bvec
def unpack(th, d, C): return {"u": (th["u"] + d[:C]).astype(np.float32), "b0": float(th["b0"] + d[C]), "w": (th["w"] + d[C + 1:]).astype(np.float32)}
out = {}
for key in (sys.argv[1:] or FAMS):
    f = Fam(key); C = len(f.parent["u"]); XG, TG = f.XT[:2000], f.TT[:2000]; XV, TV = f.XT[2000:], f.TT[2000:]
    th = {k: (np.array(v, np.float64) if k != "b0" else v) for k, v in f.parent.items()}; res = {"trained": round(f.share(f.L), 3)}
    for it in range(1, NIT + 1):
        A, bv = gn_step(f, th, XG, TG); dg = np.diag(A).copy(); best = None
        for lm in (1e-3, 1e-2, 1e-1, 1):
            d = -np.linalg.solve(A + lm * np.diag(dg + 1e-12 * dg.max()), bv)
            for al in (.5, 1.0):
                cand = unpack(th, al * d, C); kv = f.kl(f.run(XV, cand, cut=True), TV).mean()
                if best is None or kv < best[0]: best = (kv, cand, lm, al)
        th = {k: (np.array(v, np.float64) if k != "b0" else v) for k, v in best[1].items()}
        res[f"gn{it}"] = round(f.share(best[1]), 3); res[f"gn{it}_lm_al"] = [best[2], best[3]]
        print(key, it, res, f"{time.time() - t0:.0f}s", flush=True)
    cos = lambda a, b: float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    res["cos_du_gn_vs_trained"] = round(cos(th["u"] - f.parent["u"], f.L["u"] - f.parent["u"]), 2); res["cos_dw_gn_vs_trained"] = round(cos(th["w"] - f.parent["w"], f.L["w"] - f.parent["w"]), 2)
    res["trainKL_val"] = {"cut": float(f.kl(f.run(XV, cut=True), TV).mean()), "gn": float(f.kl(f.run(XV, th, cut=True), TV).mean()), "trained": float(f.kl(f.run(XV, f.L, cut=True), TV).mean())}
    np.savez(os.path.join(HERE, "figs", "l1", f"gn_{key}.npz"), u=th["u"], b0=th["b0"], w=th["w"])
    out[key] = res; print(key, "FINAL", res, flush=True)
json.dump(out, open(os.path.join(RES, "vision-n5-gn.json"), "w"), indent=1)
