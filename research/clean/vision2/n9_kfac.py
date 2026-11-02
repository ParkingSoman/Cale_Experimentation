"""N9: K-FAC closed form (shared per-block matrices, per-neuron scales) vs full GN. Rules: VISION-LOG.md N9."""
from n7_lib import *
def kfac_stats(f, th, S=2):
    gam = mx.array(GM[f.pb].astype(np.float32)); m.blocks[f.cb].fc2 = f.fc2cut; blk = m.blocks[f.pb]; C = len(th["u"])
    u = mx.array(np.asarray(th["u"], np.float32)); w = mx.array(np.asarray(th["w"], np.float32)); X_, T_ = f.XT[f.gix], f.TT[f.gix]
    G = np.zeros((C, C)); Aa = np.zeros((C, C)); gu = np.zeros(C); gb = 0.0; gw = np.zeros(C); hh2 = 0.0; ee2 = 0.0; bb = 0.0; npos = 0
    for a, b in batches(X_.shape[0], 25):
        x = X_[a:b]; pre = blk.pre(x); hall = gelu(blk.fc1(pre)) * f.mask; z = pre @ u + th["b0"]; h = gelu(z); s1 = x + blk.gamma * (blk.fc2(hall) + h[..., None] * w)
        lg = f.suffix(s1); p = mx.softmax(lg, axis=1); ps = mx.softmax(T_[a:b], axis=1); pn_ = np.array(p, np.float64); pn_ /= pn_.sum(1, keepdims=True); n_ = b - a
        dz = gelu_d(z); P2 = pre.reshape(-1, pre.shape[-1]); Aa += np.array(P2.T @ P2, np.float64); npos += P2.shape[0]
        for k, v in enumerate([p - ps] + [mx.eye(1000)[mx.array([rng.choice(1000, p=q) for q in pn_])] - p for _ in range(S)]):
            _, (g1,) = mx.vjp(f.suffix, [s1], [v]); gg = (g1 * gam).reshape(-1, C); e = ((g1 @ (gam * w)) * dz).reshape(-1)
            if k == 0:
                gw += np.array((h.reshape(-1, 1) * gg).sum(0), np.float64); gu += np.array((e[:, None] * P2).sum(0), np.float64); gb += float(e.sum())
            else:
                G += np.array(gg.T @ gg, np.float64) / S; ee2 += float((e * e).sum()) / S; eimg = e.reshape(n_, -1).sum(1); bb += float((eimg * eimg).sum()) / S
        hh2 += float((h * h).sum())
    m.blocks[f.cb].fc2 = F2[f.cb]
    return {"G": G, "Aa": Aa / npos, "hh2_mean": hh2 / npos, "ee2": ee2, "bb": bb, "gu": gu, "gb": gb, "gw": gw}
def kfac(f, th, nit=4):
    C = len(th["u"]); kv = f.trainkl(th); best_th = th; hist = []
    for it in range(nit):
        s = kfac_stats(f, th); Aw = s["G"] * s["hh2_mean"]; Au = s["Aa"] * s["ee2"]; cands = []
        for lm in (1e-2, 1e-1, 1, 10):
            dw = -np.linalg.solve(Aw + lm * np.diag(np.diag(Aw)) + 1e-12 * np.eye(C), s["gw"]); du = -np.linalg.solve(Au + lm * np.diag(np.diag(Au)) + 1e-12 * np.eye(C), s["gu"]); db = -s["gb"] / (s["bb"] * (1 + lm))
            for al in (.5, 1.0): cands.append((lm, al, {"u": th["u"] + al * du, "b0": th["b0"] + al * db, "w": th["w"] + al * dw}))
        sc = [(f.trainkl(c), lm, al, c) for lm, al, c in cands]; k2, lm, al, c = min(sc, key=lambda t: t[0])
        if k2 >= kv: break
        th, kv = c, k2; hist.append(round(f.share(th), 3))
    return th, hist
for key in (sys.argv[1:] or KEYS):
    f = load(key); z = np.load(os.path.join(CD, f"gn_{key}.npz")); gn_sh = f.share({"u": z["u"], "b0": float(z["b0"]), "w": z["w"]})
    th, hist = kfac(f, dict(f.parent)); print(key, "trained %.3f GN %.3f KFAC steps %s" % (f.share(f.L), gn_sh, hist), f"{time.time() - t0:.0f}s", flush=True)
