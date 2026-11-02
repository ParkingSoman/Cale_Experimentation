"""Cards W0 / W2: the Fisher write.
u_j = what the family gave 4a reader j (A_j . pooled members), v_j = col_j . c + b_j (what the coarse neuron gives).
Second order: KL(intact || crushed) ~ 1/2 sum_c p_c (g_c . du)^2, g_c = dlog p_c / du at the INTACT net (top classes).
W0: that prediction against real KL for existing builds (dev). W2: the write minimizing it (one linear solve, no knob,
gradients from train images). Exact case: coarse = the 10 members gives back the original columns.
usage: w_fisher.py [--ntr 1000] [--ndv 1000] [--R 10] [--exact]"""
import argparse, json, numpy as np, torch
from inception import *
from family import Family, Data, pool3, IN4A, kl
from d4_write import stats, writes

PAD = lambda x: F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=float('-inf')), 3, 1)


def upper(m, outs):
    """4a input-conv outputs (pre-ReLU readers) -> logits. Same as the tail of family.forward_family."""
    a = m.mixed4a_1x1(outs[0]); b = m.mixed4a_3x3_bottleneck(outs[1]); cc = m.mixed4a_5x5_bottleneck(outs[2])
    d = m.mixed4a_pool_reduce(outs[3])
    b = m.mixed4a_3x3(m.mixed4a_3x3_pre_relu_conv(F.pad(b, (1, 1, 1, 1))))
    cc = m.mixed4a_5x5(m.mixed4a_5x5_pre_relu_conv(F.pad(cc, (2, 2, 2, 2))))
    h = torch.cat([a, b, cc, d], 1)
    for n in ["mixed4b", "mixed4c", "mixed4d", "mixed4e"]:
        h = mixed(m, n, h)
    h = pool3(h)
    for n in ["mixed5a", "mixed5b"]:
        h = mixed(m, n, h)
    return m.softmax2_pre_activation_matmul(F.avg_pool2d(h, 7).reshape(-1, 1024))


def intact(m, fam, ins):
    """Intact reader inputs, and the family's pooled outputs (z1 for three convs, z2 for the pool branch)."""
    x = pool3(F.relu(pre_3b(m, ins[0]))); p = PAD(x)
    outs = [F.conv2d(x if j < 3 else p, getattr(m, n).weight, getattr(m, n).bias) for j, n in enumerate(IN4A)]
    return outs, x[:, fam], p[:, fam]


def fisher_grads(m, outs, R):
    """List over the top-R classes of per-image sqrt(p_c) dlog p_c / d(reader inputs); and the mass covered."""
    outs = [o.detach().requires_grad_() for o in outs]
    lp = F.log_softmax(upper(m, outs), 1); top = lp.detach().exp().topk(R, 1)
    G = []
    for r in range(R):
        w = top.values[:, r].sqrt()
        g = torch.autograd.grad((w * lp.gather(1, top.indices[:, r:r + 1])[:, 0]).sum(), outs, retain_graph=r < R - 1)
        G.append([t.detach() for t in g])
    return G, top.values.sum(1)


def upper_post(m, hs):
    """Post-ReLU outputs of the four 4a input convs -> logits (the same as upper without the first ReLUs)."""
    a, b, cc, d = hs
    b = m.mixed4a_3x3(m.mixed4a_3x3_pre_relu_conv(F.pad(b, (1, 1, 1, 1))))
    cc = m.mixed4a_5x5(m.mixed4a_5x5_pre_relu_conv(F.pad(cc, (2, 2, 2, 2))))
    h = torch.cat([a, b, cc, d], 1)
    for n in ["mixed4b", "mixed4c", "mixed4d", "mixed4e"]:
        h = mixed(m, n, h)
    h = pool3(h)
    for n in ["mixed5a", "mixed5b"]:
        h = mixed(m, n, h)
    return m.softmax2_pre_activation_matmul(F.avg_pool2d(h, 7).reshape(-1, 1024))


def fisher_grads_smooth(m, outs, R, sig):
    """Statistically linearized gradients: g_post * Phi(a / sigma) instead of g_post * 1[a > 0] (sig: list of (out_j,))."""
    hs = [F.relu(o).detach().requires_grad_() for o in outs]
    lp = F.log_softmax(upper_post(m, hs), 1); top = lp.detach().exp().topk(R, 1)
    gain = [torch.special.ndtr(o / s[None, :, None, None].clamp(min=1e-6)) for o, s in zip(outs, sig)]
    G = []
    for r in range(R):
        w = top.values[:, r].sqrt()
        g = torch.autograd.grad((w * lp.gather(1, top.indices[:, r:r + 1])[:, 0]).sum(), hs, retain_graph=r < R - 1)
        G.append([t.detach() * q for t, q in zip(g, gain)])
    return G, top.values.sum(1)


def A_cols(m, fam):
    return [getattr(m, n).weight.detach()[:, fam, 0, 0] for n in IN4A]          # (out_j, 10) each


def du(A, z1, z2, cols, cb, c1, c2):
    """u - v per conv: (B, out_j, 14, 14)."""
    res = []
    for j in range(4):
        z, c = (z1, c1) if j < 3 else (z2, c2)
        u = torch.einsum("oi,bihw->bohw", A[j], z)
        v = torch.einsum("ok,bkhw->bohw", cols[j], c) + cb[j][None, :, None, None]
        res.append(u - v)
    return res


def coarse(fm, ins):
    c1 = pool3(F.relu(fm.pre(ins))); return c1, PAD(c1)


def load_build(m, fam, tag):
    ck = torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu", weights_only=False)
    fm = Family(m, fam, ck.get("k", 1), ks=ck.get("ks", 5)).to(DEV); fm.load_state_dict(ck["state"])
    return fm


def solve(m, fam, D, fm, idx, R, k, bs=25, save=None, sig=None):
    """Normal equations for B (out x k) and b (out) per conv, all four convs jointly (one scalar residual per class)."""
    A = A_cols(m, fam); sizes = [a.shape[0] for a in A]; P = sum(sizes) * (k + 1)
    XtX = torch.zeros(P, P, dtype=torch.float64); Xty = torch.zeros(P, dtype=torch.float64); cov = []
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        with torch.no_grad():
            ins = inputs_3b(m, to_3a(m, prep(D.X[b]))); outs, z1, z2 = intact(m, fam, ins)
            c1, c2 = coarse(fm, ins) if fm is not None else (z1, z2)
        G, mass = fisher_grads(m, outs, R) if sig is None else fisher_grads_smooth(m, outs, R, sig); cov.append(mass.cpu().numpy())
        with torch.no_grad():
            for g in G:
                feats, y = [], 0.0
                for j in range(4):
                    c = c1 if j < 3 else c2; z = z1 if j < 3 else z2
                    ce = torch.cat([c, torch.ones_like(c[:, :1])], 1)                       # (B, k+1, h, w)
                    feats.append(torch.einsum("bohw,bkhw->bok", g[j], ce).reshape(len(b), -1))
                    y = y + torch.einsum("bohw,oi,bihw->b", g[j], A[j], z)
                X = torch.cat(feats, 1).cpu().double(); yy = y.cpu().double()
                XtX += X.T @ X; Xty += X.T @ yy
        print(f"  solve {i + len(b)}/{len(idx)}", flush=True) if (i // bs) % 10 == 0 else None
    if save: torch.save(dict(XtX=XtX, Xty=Xty, n=len(idx), R=R), save)
    lam = 1e-6 * torch.trace(XtX) / P
    w = torch.linalg.solve(XtX + lam * torch.eye(P, dtype=torch.float64), Xty).float()
    cols, cbs, o = [], [], 0
    for s in sizes:
        W = w[o:o + s * (k + 1)].reshape(s, k + 1); o += s * (k + 1)
        cols.append(W[:, :k].to(DEV)); cbs.append(W[:, k].to(DEV))
    return cols, cbs, float(np.concatenate(cov).mean())


def predict_and_real(m, fam, D, idx, builds, R, bs=25):
    """Per image: predicted 1/2 sum_c (g_c . du)^2 and real KL, for each build (fm, cols, cb)."""
    A = A_cols(m, fam); pred = {k: [] for k in builds}; real = {k: [] for k in builds}
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        with torch.no_grad():
            ins = inputs_3b(m, to_3a(m, prep(D.X[b]))); outs, z1, z2 = intact(m, fam, ins)
            t = upper(m, outs)
        G, _ = fisher_grads(m, outs, R)
        with torch.no_grad():
            for name, (fm, cols, cb) in builds.items():
                c1, c2 = coarse(fm, ins); d = du(A, z1, z2, cols, cb, c1, c2)
                pr = sum(sum((g[j] * d[j]).sum((1, 2, 3)) for j in range(4)) ** 2 for g in G) / 2
                pred[name].append(pr.cpu().numpy())
                real[name].append(kl(t, upper(m, [o - dj for o, dj in zip(outs, d)])).cpu().numpy())
    return {k: np.concatenate(v) for k, v in pred.items()}, {k: np.concatenate(v) for k, v in real.items()}


def real_kl(m, fam, D, idx, builds, bs=100):
    A = A_cols(m, fam); out = {k: [] for k in builds}
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; ins = inputs_3b(m, to_3a(m, prep(D.X[b]))); outs, z1, z2 = intact(m, fam, ins)
            t = upper(m, outs)
            for name, (fm, cols, cb) in builds.items():
                c1, c2 = coarse(fm, ins); d = du(A, z1, z2, cols, cb, c1, c2)
                out[name].append(kl(t, upper(m, [o - dj for o, dj in zip(outs, d)])).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def zeros_like_cols(fm):
    return [torch.zeros_like(c) for c in fm.weights()[2]], [torch.zeros(c.shape[0], device=DEV) for c in fm.weights()[2]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ntr", type=int, default=1000); ap.add_argument("--ndv", type=int, default=1000)
    ap.add_argument("--R", type=int, default=10); ap.add_argument("--exact", action="store_true")
    a = ap.parse_args()
    m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
    for p in m.parameters(): p.requires_grad_(False)
    tr = np.sort(np.random.default_rng(11).choice(D.tr, a.ntr, replace=False))
    if a.exact:   # exact case: coarse = the members themselves -> original columns, zero bias
        cols, cbs, cov = solve(m, fam, D, None, tr, a.R, 10)
        A = A_cols(m, fam)
        print("exact case: rel err per conv", [float((c - aa).norm() / aa.norm()) for c, aa in zip(cols, A)],
              "| bias norm", [float(b.norm()) for b in cbs], "| class mass", round(cov, 3), flush=True)
        raise SystemExit
    fd = load_build(m, fam, "derived2-r1-ks7"); ft = load_build(m, fam, "L-k1-r-lr3e-2")
    builds = {}
    for nm, fm in [("der", fd), ("trn", ft)]:
        res, _ = writes(m, fam, stats(m, fm, fam, D))
        for w in ["bayes", "vmr"]:
            builds[f"{nm}+{w}"] = (fm, [c.to(DEV) for c in res[w][0]], res[w][1])
    tw = ft.weights()[2]; builds["trn+trained"] = (ft, tw, [torch.zeros(c.shape[0], device=DEV) for c in tw])
    zc, zb = zeros_like_cols(fd); mu, C = stats(m, fd, fam, D)[0]
    for nm, fm in [("der", fd), ("trn", ft)]:
        print(f"W2 solve on {a.ntr} train images, read = {nm}", flush=True)
        cols, cbs, cov = solve(m, fam, D, fm, tr, a.R, 1); builds[f"{nm}+fisher"] = (fm, cols, cbs)
        print(f"  class mass covered by top-{a.R}: {cov:.3f}", flush=True)
    dv = np.sort(np.random.default_rng(5).choice(D.dv, a.ndv, replace=False))
    print(f"W0: predicted vs real on {a.ndv} dev images", flush=True)
    pred, real = predict_and_real(m, fam, D, dv, builds, a.R)
    out = {}
    for k in builds:
        r = np.corrcoef(pred[k], real[k])[0, 1]
        out[k] = dict(pred=float(pred[k].mean()), real=float(real[k].mean()), corr=float(r))
        print(f"  {k:12s} predicted {pred[k].mean():.4f}  real {real[k].mean():.4f}  per-image corr {r:.2f}", flush=True)
    # the 60 gated neurons (trained 1x1 column strongly negative) and the Fisher write's sign on them
    A = A_cols(m, fam)[0].cpu().double().numpy(); varI = np.einsum("oi,ij,oj->o", A, C[:10, :10], A)
    sc = np.sqrt(C[10, 10]); tcol = tw[0][:, 0].cpu().numpy(); gated = tcol * sc < -np.sqrt(varI)
    for nm in ["der+fisher", "trn+fisher"]:
        fcol = builds[nm][1][0][:, 0].cpu().numpy()
        print(f"  gate: {gated.sum()} gated 1x1 neurons; {nm} column negative on {(fcol[gated] < 0).sum()}; "
              f"corr(trained col, {nm} col) over 192 = {np.corrcoef(tcol, fcol)[0, 1]:.2f}", flush=True)
    json.dump(out, open("../../../results/curves-w0w2.json", "w"), indent=1)
    torch.save({k: ([c.cpu() for c in v[1]], [b.cpu() for b in v[2]]) for k, v in builds.items() if "fisher" in k}, "cache/fisher_writes.pt")
