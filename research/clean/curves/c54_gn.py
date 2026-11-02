"""C4 (Part VIII, 31.9): preconditioned nonlinear CG on L_ex (exact reader gates, uniform output metric). SMALL test.
usage: c54_gn.py SOURCE NTRAIN TAG   (SOURCE = photos | path to network-made .npy)"""
import numpy as np, torch, time, sys
SOURCE, NTR, TAG = sys.argv[1], int(sys.argv[2]), sys.argv[3]
PREC = sys.argv[4] if len(sys.argv) > 4 else "kfc"
_keep = (SOURCE, NTR, TAG, PREC) if "PREC" in dir() else None
sys.argv = ["x", "photos", "300", TAG]
SRC = open("c50_exactcg.py").read(); exec(SRC.split("# self-check of Prop. 31.2")[0])
def passes(V, X, want_grad=True, Hd=None, bs=10):
    """L_ex(V), grad L_ex(V) (current gates), and optionally d^T H_gated d for direction Hd, over X."""
    L = 0.0; Gr = torch.zeros(NRD, 962, device=DEV); curv = 0.0
    for i in range(0, len(X), bs):
        x, p = fin(X[i:i + bs]); o = torch.ones_like(x[:, :1]); y = torch.cat([x, o, p, o], 1).flatten(2)
        with torch.no_grad():
            pre = [F.conv2d(inp, getattr(m, n).weight, getattr(m, n).bias) for n, inp in zip(IN4A, [x, x, x, p])]
            a = torch.cat([t.flatten(2) for t in pre], 1); e = torch.einsum("jk,bkp->bjp", V * mask, y)
            dl = F.relu(a - e) - F.relu(a); Dg = (a - e > 0).float()
        pp = [F.relu(t).clone().requires_grad_(True) for t in pre]; lg = upper(*pp)
        w = torch.zeros_like(lg, requires_grad=True); gw = torch.autograd.grad((lg * w).sum(), pp, create_graph=True)
        def Jv(t3):
            tans = [t.reshape(len(x), -1, 14, 14) for t in torch.split(t3, sizes, 1)]
            u = torch.autograd.grad(sum((g_ * t).sum() for g_, t in zip(gw, tans)), w, retain_graph=True)[0].detach(); return u - u.mean(1, keepdim=True)
        u = Jv(dl); L += 0.5 * float((u ** 2).sum())
        if want_grad:
            gt = torch.cat([g.flatten(2) for g in torch.autograd.grad((lg * u).sum(), pp, retain_graph=True)], 1)
            Gr -= torch.einsum("bjp,bkp->jk", Dg * gt, y) * mask
        if Hd is not None:
            ud = Jv(Dg * torch.einsum("jk,bkp->bjp", Hd * mask, y)); curv += float((ud ** 2).sum())
    n = len(X); return L / n, Gr * free / n, curv / n
if PREC == "centered":                                   # Prop. 31.10: centered KFC from the same G, A
    A64 = Af.cpu().double(); mu = torch.zeros(962, dtype=torch.float64); mu[:480] = A64[:480, 480]; mu[481:961] = A64[481:961, 961]
    Cm = torch.eye(962, dtype=torch.float64); Cm[:480, 480] = mu[:480]; Cm[481:961, 961] = mu[481:961]
    Ci = torch.eye(962, dtype=torch.float64); Ci[:480, 480] = -mu[:480]; Ci[481:961, 961] = -mu[481:961]
    Ap = Ci @ A64 @ Ci.T; Api = torch.linalg.inv(Ap + 1e-6 * Ap.trace() / 962 * torch.eye(962, dtype=torch.float64))
    Mright = (Cm @ Api @ Ci).float().to(DEV)
    Pinv = lambda R: (Gi @ (R * free) @ Mright) * free
    print(f"centered KFC: A const diag {float(A64[480,480]):.3g}, Cov median diag {float(Ap.diag()[:480].median()):.3g}", flush=True)
rng = np.random.default_rng(50); perm = rng.permutation(D.tr)
if SOURCE == "photos": Xtr = D.X[np.sort(perm[:NTR])]
else: A = np.load(SOURCE); Xtr = A[np.sort(np.random.default_rng(54).permutation(len(A))[:NTR])]
Xho = D.X[np.sort(perm[2000:2300])]; Idv = np.sort(np.random.default_rng(54).choice(D.dv, 1000, replace=False))
refs = REFS(); S0 = Vs["S"] * mask; T0 = Vs["T"] * mask; dTn = float(((T0 - S0) * free).norm())
kS = per_image(build(S0), Idv); print(f"S on dev-1000: {kS.mean():.4f}", flush=True)
v = S0.clone(); L, g, _ = passes(v, Xtr); z = Pinv(g); d = -z; gz = float((g * z).sum())
for k in range(1, 13):
    _, _, c = passes(v, Xtr, want_grad=False, Hd=d); a = -float((g * d).sum()) / c
    for h in range(5):
        Ln, gn, _ = passes(v + a * d, Xtr)
        if Ln < L: break
        a /= 2
    v = v + a * d; L = Ln; zn = Pinv(gn); gzn = float((gn * zn).sum()); beta = max(0.0, (gzn - float((gn * z).sum())) / gz)
    d = -zn + beta * d; g, z, gz = gn, zn, gzn
    kd = per_image(build(v), Idv); Lh = passes(v, Xho, want_grad=False)[0]
    print(f"  it {k}: train L_ex {L:.4g}  held-out L_ex {Lh:.4g}  dev-1000 KL {kd.mean():.4f} (vs S {float((kd - kS).mean()):+.4f}, se {float((kd - kS).std() / np.sqrt(1000)):.4f})  |v-S|/|T-S| {float(((v - S0) * free).norm()) / dTn:.1f}  halvings {h} beta {beta:.2f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(v.cpu(), f"cache/c54_{TAG}_it{k}_V.pt")
print("done")
