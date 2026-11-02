"""F0 + F2 (Part VII): reader-layer Fisher F = E_x sum_c vec(Gamma_c) vec(Gamma_c)^T, Gamma_c(x) = sum_p g_c(x,p) y(x,p)^T
(g at reader OUTPUTS, centred and sqrt(p)-weighted over the top-10 classes; y = (x-activity, 1, p-activity, 1), 962).
Objectives for each build's weight change V (368 x 962, masked): L_img (sampled F), plain KFC (G (x) A), EKFAC (s_mn in
the Kronecker eigenbasis). F0: gradient norm of L_img at T vs S on the free coordinates."""
import numpy as np, torch, time
src = open("c11_exactavg.py").read(); exec(src.split("Lf = {k: 0.0")[0].replace(
    "G = torch.cat([reader_grads(Xm[i:i + 20]).flatten(1).half() for i in range(0, 500, 20)])", "G = None").replace('print(f"G {tuple(G.shape)}', 'print(f"G skipped'))
builds["4d"] = load_state("cache/c8_build_4d.pt")
NRD = 368; nx = sum(sizes[:3]); famt = torch.tensor(fam); restt = torch.tensor(keep + [480])
mask = torch.zeros(NRD, 962, device=DEV); mask[:nx, :481] = 1; mask[nx:, 481:] = 1
free = mask.clone(); free[:nx, famt] = 0; free[nx:, famt + 481] = 0
def Vmat(fm):
    cs = [float(c) for c in fm.cscale]; V = torch.zeros(NRD, 962, device=DEV); off = np.cumsum([0] + sizes)
    for j in range(4):
        Bj = torch.cat([fm.dW[j].detach() * cs[j], fm.cbp[j].detach()[:, None] * 0.1], 1); o = 0 if j < 3 else 481; rows = slice(off[j], off[j + 1])
        V[rows, o + famt.to(DEV)] = WF[j]; V[rows, o + restt.to(DEV)] = -Bj
    return V
Vs = {k: Vmat(fm) for k, fm in builds.items()}
def ybatch(b):
    x, p = family_inputs(b); o = torch.ones_like(x[:, :1]); return torch.cat([x, o, p, o], 1).flatten(2)        # (B, 962, 196)
def gammas(b):
    g = reader_grads(b).reshape(len(b), 10, NRD, 196); y = ybatch(b)
    return torch.einsum("bcjp,bkp->bcjk", g, y) * mask, g, y                                                    # (B, 10, 368, 962)
G = torch.zeros(NRD, NRD, device=DEV); A = torch.zeros(962, 962, device=DEV); Limg = {k: 0.0 for k in Vs}
grad = {k: torch.zeros(NRD, 962, device=DEV) for k in ["S", "T"]}
for i in range(0, 2000, 10):
    b = TR[i:i + 10]; Gm, g, y = gammas(b)
    G += torch.einsum("bcjp,bckp->jk", g, g); A += torch.einsum("bjp,bkp->jk", y, y) / 196
    if i < 500:
        for k, V in Vs.items():
            sc = (Gm * V).sum((2, 3)); Limg[k] += float((sc ** 2).sum()) / 2
            if k in grad: grad[k] += torch.einsum("bc,bcjk->jk", sc, Gm)
G /= 2000; A /= 2000; Limg = {k: v / 500 for k, v in Limg.items()}
lg, UG = torch.linalg.eigh(G.cpu().double()); la, UA = torch.linalg.eigh(A.cpu().double()); UG, UA = UG.float().to(DEV), UA.float().to(DEV)
print(f"factors done ({time.time() - t0:.0f}s)", flush=True)
s = torch.zeros(NRD, 962, device=DEV); s256 = None
for i in range(0, 2000, 10):
    Gm, _, _ = gammas(TR[i:i + 10]); P = torch.einsum("jm,bcjk,kn->bcmn", UG, Gm, UA); s += (P ** 2).sum((0, 1))
    if i + 10 == 250: s256 = s.clone() / 250
s /= 2000
lgd, lad = torch.from_numpy(lg.numpy()).float().to(DEV), torch.from_numpy(la.numpy()).float().to(DEV)
kron = lgd[:, None].clamp(min=0) * lad[None, :].clamp(min=0)
def L_kfc(V): return 0.5 * float((V * (G @ V @ A)).sum())
def L_e(V, s_): return 0.5 * float((s_ * (UG.T @ V @ UA) ** 2).sum())
print(f"{'build':10s} {'L_img':>9s} {'L_KFC':>9s} {'L_EKFAC':>9s} {'L_E(256)':>9s}")
rows = {k: (Limg[k], L_kfc(V), L_e(V, s), L_e(V, s256)) for k, V in Vs.items()}
for k, r in rows.items(): print(f"{k:10s} " + " ".join(f"{v:9.4f}" for v in r))
for col, name in enumerate(["L_img", "L_KFC", "L_EKFAC", "L_E(256)"]):
    print(f"  {name}: S+rem/S {rows['S+rem'][col] / rows['S'][col]:.3f}   T/S {rows['T'][col] / rows['S'][col]:.3f}   c7_reader/S {rows['c7_reader'][col] / rows['S'][col]:.3f}")
gS, gT = grad["S"] * free / 500, grad["T"] * free / 500
print(f"F0: |grad_free L_img(T)| / |grad_free L_img(S)| = {float(gT.norm() / gS.norm()):.3f}")
rem = (Vs["T"] - Vs["S"]) * free
print(f"    cos(-grad L_img(S), T - S) on free coords = {float(-(gS * rem).sum() / gS.norm() / rem.norm()):.3f}")
rho = s / (kron + 1e-12 * kron.max()); top = torch.argsort((s).flatten(), descending=True)[:1000]
import scipy.stats as ss
s256r = (s256 / (kron + 1e-12 * kron.max())).flatten()[top].cpu().numpy(); s2000r = rho.flatten()[top].cpu().numpy()
print(f"E-small: Spearman of rho over top-1000 s pairs, 256 vs 2000 images: {ss.spearmanr(s256r, s2000r).correlation:.3f}")
torch.save(dict(G=G.cpu(), A=A.cpu(), s=s.cpu(), s256=s256.cpu(), UG=UG.cpu(), UA=UA.cpu(), lg=lg, la=la), "cache/c17_fisher.pt")
print(f"done ({time.time() - t0:.0f}s)")
