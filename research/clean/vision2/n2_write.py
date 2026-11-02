"""N2: closed-form write nudge by weighted least squares. Rules: VISION-LOG.md N2."""
from n_harness import *
rng = np.random.default_rng(3); NS = 4
def suffix_fn(f, start, end_fn):
    """function from the stream at block `start` OUTPUT (i.e. input of start+1) to end_fn(h_at_end)."""
    def fn(h):
        for i in range(start + 1, NB):
            if ST[i] != ST[i - 1]: h = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](h))
            h = m.blocks[i](h)
            r = end_fn(i, h)
            if r is not None: return r
        return m.head(h)
    return fn
out = {}
for key in FAMS:
    f = Fam(key); pb, cb = f.pb, f.cb; m.blocks[cb].fc2 = f.fc2cut
    hp_all, hc_all, rows = [], [], {"M0": None, "M1": [], "M2": []}
    feats = {"M1": [], "M2": []}; targs = {"M1": [], "M2": []}
    gp = GM[pb]; gc = GM[cb]; wcs = np.stack(f.wc, 1) * gc[:, None]      # child writes in stream units (C, k)
    A0 = []; T0 = []
    for a, b in batches(f.XT.shape[0], 25):
        x = f.XT[a:b]; blk = m.blocks[pb]; pre = blk.pre(x); hp = gelu(blk.fc1(pre)[..., f.pn]); s1 = blk(x)          # s1: stream at pb output (intact)
        m.blocks[cb].fc2 = F2[cb]; s2i = m.blocks[cb](s1); hc = gelu(m.blocks[cb].fc1(m.blocks[cb].pre(s1))[..., mx.array([c[1] for c in f.cs])]); m.blocks[cb].fc2 = f.fc2cut
        s2 = m.blocks[cb](s1)                                                                                            # stream at cb output, children cut
        hpn = np.array(hp).reshape(b - a, -1).astype(np.float64); hcn = (np.array(hc).reshape(b - a, -1, len(f.cs)) - f.mu).astype(np.float64)
        lost = np.einsum("npk,ck->npc", hcn, wcs)                                                                         # (n, P, C) children's lost write
        A0.append(hpn.reshape(-1)); T0.append(lost.reshape(-1, lost.shape[-1]))
        for M in ("M1", "M2"):
            for smp in range(NS):
                if M == "M2":
                    lg = m.head(suffix_fn(f, cb, lambda i, h: None)(s2)) if False else None
                    f2 = suffix_fn(f, cb, lambda i, h: None); lg = f2(s2); p = mx.softmax(lg, axis=1)
                    yy = mx.array([rng.choice(1000, p=np.array(p[j], np.float64) / np.array(p[j], np.float64).sum()) for j in range(b - a)])
                    v = mx.eye(1000)[yy] - p
                    _, (g2,) = mx.vjp(f2, [s2], [v])
                    f1 = lambda h: f2(m.blocks[cb](h)); _, (g1,) = mx.vjp(f1, [s1], [v])
                else:   # child-local: next block's neuron pre-activations
                    nb = cb + 1
                    endf = lambda i, h: m.blocks[i].fc1(m.blocks[i].pre(h)) if i == nb else None
                    def f2(h):
                        if nb >= NB: return m.head(h)
                        if ST[nb] != ST[cb]: h = m.down[ST[nb] - 1](m.down_norm[ST[nb] - 1](h))
                        return m.blocks[nb].fc1(m.blocks[nb].pre(h))
                    zz = f2(s2); v = mx.array(rng.standard_normal(zz.shape).astype(np.float32))
                    _, (g2,) = mx.vjp(f2, [s2], [v]); f1 = lambda h: f2(m.blocks[cb](h)); _, (g1,) = mx.vjp(f1, [s1], [v])
                g1n = np.array(g1).reshape(b - a, -1, g1.shape[-1]).astype(np.float64); g2n = np.array(g2).reshape(b - a, -1, g2.shape[-1]).astype(np.float64)
                if g1n.shape[1] != hpn.shape[1]: continue
                feats[M].append(np.c_[np.einsum("np,npc->nc", hpn, g1n), g1n.sum(1)])                  # unknowns: dw (C), bias (C), stream units
                targs[M].append(np.einsum("npc,npc->n", g2n, lost))
    m.blocks[cb].fc2 = F2[cb]
    sols = {}
    hpv = np.concatenate(A0); Tv = np.concatenate(T0); hm = hpv.mean(); hc_ = hpv - hm
    dw0 = (hc_ @ (Tv - Tv.mean(0))) / (hc_ @ hc_); sols["M0"] = (dw0, Tv.mean(0) - dw0 * hm)
    Lm = Tv.mean(0)
    for M in ("M1", "M2"):
        Fm = np.concatenate(feats[M]); tm = np.concatenate(targs[M]); C = Fm.shape[1] // 2
        # bias fixed by mean matching: bias = Lm - dw*hm  ->  row: (Fh - hm*Fs).dw = t - Fs.Lm
        Fx = Fm[:, :C] - hm * Fm[:, C:]; tx = tm - Fm[:, C:] @ Lm
        nr_ = len(tx); va = np.arange(nr_) % 5 == 0; trr = ~va; best = None
        G0 = Fx[trr].T @ Fx[trr]; sc = np.trace(G0) / C
        for lam in (1e-3, 1e-2, 1e-1, 1, 10):
            d = np.linalg.solve(G0 + np.eye(C) * lam * sc, Fx[trr].T @ tx[trr]); e = ((Fx[va] @ d - tx[va]) ** 2).sum()
            if best is None or e < best[0]: best = (e, lam)
        d = np.linalg.solve(Fx.T @ Fx + np.eye(C) * best[1] * np.trace(Fx.T @ Fx) / C, Fx.T @ tx); sols[M] = (d, Lm - d * hm); print("  ", M, "ridge", best[1], flush=True)
    res = {}
    XV = f.XT[:1000]; TV = f.TT[:1000]; kv0 = f.kl(f.run(XV, cut=True), TV).mean()
    for M, (dw, db) in sols.items():
        bestA = (np.inf, 1.0)
        for al in (.25, .5, 1, 1.5):
            nr = {"u": f.parent["u"], "b0": f.parent["b0"], "w": (f.parent["w"] + al * dw / gp).astype(np.float32)}
            kv = f.kl(f.run(XV, nr, (al * db / gp + (1 - al) * 0).astype(np.float32), cut=True), TV).mean()
            if kv < bestA[0]: bestA = (kv, al)
        al = bestA[1]; nr = {"u": f.parent["u"], "b0": f.parent["b0"], "w": (f.parent["w"] + al * dw / gp).astype(np.float32)}
        res[M] = round(f.share(nr, (al * db / gp).astype(np.float32)), 3); res[M + "_step"] = al
        res[M + "_cos_learned_dw"] = round(float(np.dot(dw / gp, f.L["w"] - f.parent["w"]) / np.linalg.norm(dw / gp) / np.linalg.norm(f.L["w"] - f.parent["w"])), 2)
    out[key] = res; print(key, res, f"{time.time() - t0:.0f}s", flush=True)
json.dump(out, open(os.path.join(RES, "vision-n2-write.json"), "w"), indent=1)
