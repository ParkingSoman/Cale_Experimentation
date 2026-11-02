"""C31: reader-coverage completion. Feature visualisations of the silent 4a readers (plain pixels), added to the sets."""
import sys, numpy as np, torch
from lucent.optvis import param, transform
from inception import *
from family import Data, pool3
import w_fisher as W
from w_fisher import load_build, real_kl, intact
import lucent.optvis.param.spatial as _sp, lucent.optvis.param.color as _co, lucent.optvis.transform as _tr
for _mod in (_sp, _co, _tr): _mod.device = DEV
m = load_model(); mr = load_model(redirected=True); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"]); M = load_build(m, fam, "L-k3-m-lr3e-2")
for p in m.parameters(): p.requires_grad_(False)
class Src:
    def __init__(self, X): self.X = X
INV = np.load("cache/e1_self_256.npy").astype(np.float32); yinv = np.load("cache/e1_self_256_y.npy")
Wl = m.softmax2_pre_activation_matmul.weight.detach()[torch.tensor(yinv)]; Wn = Wl / Wl.norm(dim=1, keepdim=True); S = (Wn @ Wn.T).cpu().numpy(); np.fill_diagonal(S, -2); order = np.argsort(-S, 1)
B4 = np.stack([INV[np.r_[[a], order[a, r]]].mean(0) for r in [[0, 1, 2], [1, 2, 3]] for a in range(256)]).clip(0, 255).astype(np.uint8); B4 = np.concatenate([B4, B4[:, :, ::-1]])
BND = np.load("cache/e7_bnd_1024_128_0.npy"); BND = np.concatenate([BND, BND[:, :, ::-1]])
def onrate(X, net=m):
    acc = 0; n = 0
    with torch.no_grad():
        for i in range(0, len(X), 32):
            outs, _, _ = intact(net, fam, inputs_3b(net, to_3a(net, prep(X[i:i + 32])))); acc = acc + torch.cat([(o > 0).float().mean((0, 2, 3)) * len(o) for o in outs]); n += len(outs[0])
    return (acc / n).cpu().numpy()
from scipy.special import softmax
P = softmax(np.asarray(D.T[D.tr], np.float64), 1); uns = D.tr[1 - (P ** 2).sum(1) > .1]; ridx = np.sort(np.random.default_rng(91).choice(uns, 256, replace=False))
on_r = onrate(np.asarray(D.X[ridx])); silent = np.where((onrate(BND[:512]) < .1 * on_r) | (onrate(B4[:512]) < .1 * on_r))[0]; print(f"silent readers: {len(silent)}", flush=True)
tf = transform.compose(transform.standard_transforms); FV = []
for seed in [0, 1]:
    for i in range(0, len(silent), 32):
        rs = torch.tensor(silent[i:i + 32], device=DEV); torch.manual_seed(seed * 1000 + i)
        params, img_f = param.image(224, batch=len(rs), fft=False, decorrelate=False); opt = torch.optim.Adam(params, lr=0.05)
        for s in range(128):
            outs, _, _ = intact(mr, fam, inputs_3b(mr, to_3a(mr, tf(img_f()) * 255.0 - 117.0))); U = torch.cat(outs, 1).mean((2, 3))
            loss = -U.gather(1, rs[:, None]).sum(); opt.zero_grad(); loss.backward(); opt.step()
        FV.append((img_f().detach().permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)); print(f"  fv seed {seed} {i + len(rs)}/{len(silent)}", flush=True)
FV = np.concatenate(FV); FV = np.concatenate([FV, FV[:, :, ::-1]]); np.save("cache/c31_fv.npy", FV)
on_fv = onrate(FV); print(f"silent readers now on (> 10% of real) in the reader inputs: {int((on_fv[silent] >= .1 * on_r[silent]).sum())} of {len(silent)}")
Nr = torch.load("cache/e10_N_real.pt"); Nr = Nr["XtX"] / Nr["n"]; Pn = Nr.shape[0]; out = {}
for k, X in [("boundary + readers", np.concatenate([BND, FV])), ("4-way + readers", np.concatenate([B4, FV]))]:
    c0, b0, _ = W.solve(m, fam, Src(X), M, np.arange(len(X)), 10, 3, save=f"cache/c31_N.pt"); out[k] = (M, c0, b0)
    d = torch.load("cache/c31_N.pt"); Nq = d["XtX"] / d["n"]; L = torch.linalg.cholesky(Nq + 1e-6 * torch.trace(Nq) / Pn * torch.eye(Pn, dtype=Nq.dtype)); Li = torch.linalg.inv(L)
    ev = torch.linalg.eigvalsh(Li @ Nr @ Li.T).flip(0); print(f"{k} ({len(X)}): generalised eigenvalues top {float(ev[0]):.3e}, 10th {float(ev[9]):.3e}, 100th {float(ev[99]):.3e}", flush=True)
K = real_kl(m, fam, D, D.dv, out)
for k, v in K.items(): print(f"  step0 from {k}: dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
