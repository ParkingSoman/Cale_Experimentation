"""N3: family-local metric (the family's own readers) + read nudge. Rules: VISION-LOG.md N3."""
from n_harness import *
import sys
METRIC = sys.argv[1] if len(sys.argv) > 1 else "family"; rng = np.random.default_rng(4); NS = 4; LAMS = (0, .1, .2, .4); K = 24
OFF = np.concatenate([[0], np.cumsum(H)]); EDG = np.load(os.path.join(HF, "imagenet", "e1_edges.npz")); EB = {}
def readers(b, n, k=K):
    sc = []
    for b2 in range(b + 1, NB):
        if b2 not in EB: EB[b2] = np.abs(EDG[str(b2)])
        e = EB[b2][:, OFF[b] + n]; sc += [(float(v), b2, int(j)) for j, v in enumerate(e)]
    sc.sort(reverse=True); return [(b2, j) for _, b2, j in sc[:k]]
out = {}
for key in (sys_keys := FAMS):
    f = Fam(key); pb, cb = f.pb, f.cb
    R = set(readers(pb, f.pn));
    for c in f.cs: R |= set(readers(*c))
    R = sorted(R); RB = sorted({b for b, _ in R}); RJ = {b: mx.array([j for bb, j in R if bb == b]) for b in RB}
    def fread(h, start):        # stream at block `start` OUTPUT -> concat of readers' pre-activations (+ head logits if the family is at the end)
        outs = []
        for i in range(start + 1, NB):
            if ST[i] != ST[i - 1]: h = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](h))
            if i in RJ: z = m.blocks[i].fc1(m.blocks[i].pre(h))[..., RJ[i]]; outs.append(z.reshape(z.shape[0], -1))
            if i >= max(RB + [cb]): break
            h = m.blocks[i](h)
        if cb == NB - 1 or not outs or start == NB - 1: outs.append(m.head(h if start < NB - 1 else h))
        return mx.concatenate(outs, axis=1)
    def fread1(h):              # from the parent's write point: readers inside block cb, then everything after
        o = []
        if cb in RJ: z = m.blocks[cb].fc1(m.blocks[cb].pre(h))[..., RJ[cb]]; o.append(z.reshape(z.shape[0], -1))
        o.append(fread(m.blocks[cb](h), cb)); return mx.concatenate(o, axis=1)
    def fread2(h):              # from the children's write point: block-cb readers do not see it
        o = []
        if cb in RJ: o.append(mx.zeros((h.shape[0], h.shape[1] * h.shape[2] * len(RJ[cb]))))
        o.append(fread(h, cb)); return mx.concatenate(o, axis=1)
    if METRIC == "fisher":
        def fread2(h):
            for i in range(cb + 1, NB):
                if ST[i] != ST[i - 1]: h = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](h))
                h = m.blocks[i](h)
            return m.head(h)
        def fread1(h): return fread2(m.blocks[cb](h))
    P, zp, zc = f.acts(f.XT, 1000); P1 = np.c_[P, np.ones(len(zp))]; Gp = P1.T @ P1; Gp += np.eye(len(Gp)) * 1e-3 * np.trace(Gp) / len(Gp)
    Uc = np.linalg.solve(Gp, P1.T @ zc); vote = P1 @ Uc; Uv = Uc.sum(1) * (zp.std() / max(vote.sum(1).std(), 1e-6))     # summed child votes, scaled to z_p's spread
    uv, bv = Uv[:-1].astype(np.float32), float(Uv[-1]); vmean = float((P1 @ Uv).mean())
    m.blocks[cb].fc2 = f.fc2cut; gc = GM[cb]; gp = GM[pb]; wcs = np.stack(f.wc, 1) * gc[:, None]; wps = f.parent["w"] * gp
    feats = {l: [] for l in LAMS}; targs = {l: [] for l in LAMS}; hsum = {l: [] for l in LAMS}; lostm = []; hpm = []
    for a, b in batches(f.XT.shape[0], 25):
        x = f.XT[a:b]; blk = m.blocks[pb]; pre = blk.pre(x); zpp = blk.fc1(pre)[..., f.pn]; hp = gelu(zpp); s1 = blk(x)
        m.blocks[cb].fc2 = F2[cb]; hc = gelu(m.blocks[cb].fc1(m.blocks[cb].pre(s1))[..., mx.array([c[1] for c in f.cs])]); m.blocks[cb].fc2 = f.fc2cut
        s2 = m.blocks[cb](s1); n_ = b - a
        hpn = np.array(hp).reshape(n_, -1).astype(np.float64); hcn = (np.array(hc).reshape(n_, -1, len(f.cs)) - f.mu).astype(np.float64)
        lost = np.einsum("npk,ck->npc", hcn, wcs); lostm.append(lost.reshape(-1, lost.shape[-1])); hpm.append(hpn.reshape(-1))
        vt = np.array(pre @ mx.array(uv) + bv - vmean).reshape(n_, -1).astype(np.float64); zpn = np.array(zpp).reshape(n_, -1).astype(np.float64)
        hs = {l: np.array(gelu(mx.array((zpn + l * vt).astype(np.float32)))).astype(np.float64) for l in LAMS}
        for l in LAMS: hsum[l].append(hs[l].reshape(-1))
        for smp in range(NS):
            if METRIC == "fisher":
                lg = fread2(s2); p = mx.softmax(lg, axis=1); pn_ = np.array(p, np.float64); pn_ /= pn_.sum(1, keepdims=True)
                v = mx.eye(1000)[mx.array([rng.choice(1000, p=q) for q in pn_])] - p
            else:
                zz = fread1(s1); v = mx.array(rng.standard_normal(zz.shape).astype(np.float32))
            _, (g2,) = mx.vjp(fread2, [s2], [v]); _, (g1,) = mx.vjp(fread1, [s1], [v])
            g1n = np.array(g1).reshape(n_, -1, g1.shape[-1]).astype(np.float64); g2n = np.array(g2).reshape(n_, -1, g2.shape[-1]).astype(np.float64)
            tl = np.einsum("npc,npc->n", g2n, lost); par = np.einsum("np,npc->n", hpn, g1n @ wps[:, None] if False else np.einsum("npc,c->np", g1n, wps)[..., None].repeat(1, -1)) if False else np.einsum("np,np->n", hpn, np.einsum("npc,c->np", g1n, wps))
            gs = g1n.sum(1)
            for l in LAMS:
                feats[l].append(np.c_[np.einsum("np,npc->nc", hs[l], g1n), gs]); targs[l].append(tl + par)
    m.blocks[cb].fc2 = F2[cb]
    Lm = np.concatenate(lostm).mean(0); hpmean = np.concatenate(hpm).mean(); XV = f.XT[:1000]; TV = f.TT[:1000]; res = {"n_readers": len(R)}; best = None
    for l in LAMS:
        Fm = np.concatenate(feats[l]); tm = np.concatenate(targs[l]); C = Fm.shape[1] // 2; hm = np.concatenate(hsum[l]).mean()
        # unknown w' (stream units); bias by mean matching: bias = Lm + wps*hpmean - w'*hm
        Fx = Fm[:, :C] - hm * Fm[:, C:]; tx = tm - Fm[:, C:] @ (Lm + wps * hpmean)
        va = np.arange(len(tx)) % 5 == 0; G0 = Fx[~va].T @ Fx[~va]; sc = np.trace(G0) / C; bl = None
        for lam in (1e-3, 1e-2, 1e-1, 1, 10):
            d = np.linalg.solve(G0 + np.eye(C) * lam * sc, Fx[~va].T @ tx[~va] + lam * sc * wps); e = ((Fx[va] @ d - tx[va]) ** 2).sum()   # ridge toward w_p
            if bl is None or e < bl[0]: bl = (e, lam)
        w_ = np.linalg.solve(Fx.T @ Fx + np.eye(C) * bl[1] * np.trace(Fx.T @ Fx) / C, Fx.T @ tx + bl[1] * np.trace(Fx.T @ Fx) / C * wps); bias = Lm + wps * hpmean - w_ * hm
        nr = {"u": (f.parent["u"] + l * uv).astype(np.float32), "b0": float(f.parent["b0"] + l * (bv - vmean)), "w": (w_ / gp).astype(np.float32)}
        kv = f.kl(f.run(XV, nr, (bias / gp).astype(np.float32), cut=True), TV).mean()
        sh = f.share(nr, (bias / gp).astype(np.float32)); res[f"lam{l}"] = round(sh, 3)
        if best is None or kv < best[0]: best = (kv, l, sh, nr, bias / gp)
    res["chosen_lam"] = best[1]; res["M3"] = round(best[2], 3)
    np.savez(os.path.join(HERE, "figs", "l1", f"cf_{METRIC}_{key}.npz"), u=best[3]["u"], b0=best[3]["b0"], w=best[3]["w"], dbias=best[4]); out[key] = res; print(key, res, f"{time.time() - t0:.0f}s", flush=True)
json.dump(out, open(os.path.join(RES, "vision-n3-family.json" if METRIC == "family" else "vision-n4-fisher-read.json"), "w"), indent=1)
