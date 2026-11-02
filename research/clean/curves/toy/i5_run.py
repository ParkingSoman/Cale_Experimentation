"""I5: generate with G0, G1, G2 from the toy network; compare with the truth; P2 proxy (delete half of conv2, refit the write)."""
import sys, time, numpy as np, torch, torch.nn.functional as F
from shapes import DEV, Net, sample
from gens import g0_inversion, g1_langevin, g2_stationary
torch.manual_seed(0); ck = torch.load("net_1000.pt"); net = Net().to(DEV); net.load_state_dict(ck["state"]); net.eval()
for p in net.parameters(): p.requires_grad_(False)
Xtr, ytr = ck["X"].to(DEV), ck["y"].to(DEV); M = 1000; mult = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
Xtruth, _ = sample(M, 7); Xeval, _ = sample(5000, 8)
sets = {"truth (fresh)": Xtruth, "train set": Xtr}; t0 = time.time()
import os
if "reuse" in sys.argv and os.path.exists(f"gen_x{mult:g}.pt"):
    gz = torch.load(f"gen_x{mult:g}.pt"); x2, c2 = gz["G2 stationary"].to(DEV), gz["c2"].to(DEV)
    for k in ["G0 inversion", "G1 Langevin", "G2 stationary"]: sets[k] = gz[k].to(DEV)
else:
    sets["G0 inversion"] = g0_inversion(net, M, steps=int(256 * mult))[0]; print(f"G0 done {time.time() - t0:.0f} s", flush=True)
    sets["G1 Langevin"] = g1_langevin(net, M, steps=int(1000 * mult))[0]; print(f"G1 done {time.time() - t0:.0f} s", flush=True)
    x2, y2, c2 = g2_stationary(net, M, steps=int(2000 * mult), log=print); sets["G2 stationary"] = x2; print(f"G2 done {time.time() - t0:.0f} s", flush=True)
    torch.save({k: v.cpu() for k, v in sets.items()} | {"c2": c2.cpu()}, f"gen_x{mult:g}.pt")
W = {}
if "g2b" in sys.argv:   # G2 started from G1's unsure samples: refine x and c, or reweight only (c)
    x1 = sets["G1 Langevin"]
    xb, _, cb = g2_stationary(net, M, steps=2000, lr=0.05, x_init=x1, log=print); sets["G2 from G1, refine"] = xb; W["G2 from G1, refine"] = cb
    _, _, cw = g2_stationary(net, M, steps=2000, lr=0.05, x_init=x1, fix_x=True, log=print); sets["G2 from G1, reweight"] = x1; W["G2 from G1, reweight"] = cw
    for k, c in W.items(): print(f"  {k}: effective sample size of c = {float(1 / (c ** 2).sum()):.0f} of {M}")
# statistics
def feats(X): return net.feats(X)                                         # (n, 32, 8, 8)
def unc(X): p = F.softmax(net(X), 1); return float((1 - (p ** 2).sum(1)).mean())
ft = feats(Xtruth).permute(1, 0, 2, 3).reshape(32, -1); Rt = torch.corrcoef(ft); ont = (ft > 0).float().mean(1); iu = torch.triu_indices(32, 32, 1)
print(f"\n{'set':16s} | 1-sum p^2 | co-firing corr vs truth | on-rate ratio (median, range)")
for k, X in sets.items():
    f = feats(X).permute(1, 0, 2, 3).reshape(32, -1); R = torch.corrcoef(f); on = (f > 0).float().mean(1)
    cc = float(torch.corrcoef(torch.stack([torch.nan_to_num(R[iu[0], iu[1]]), Rt[iu[0], iu[1]]]))[0, 1]); r = (on / ont.clamp(min=1e-6))
    print(f"{k:16s} | {unc(X):.3f}     | {cc:+.3f}                  | {float(r.median()):.2f} ({float(r.min()):.2f}-{float(r.max()):.2f})")
# I5a: does G2 recover training images?
Xt = Xtr.flatten(1); Xt = (Xt - Xt.mean(1, keepdim=True)) / Xt.std(1, keepdim=True)
G = x2.flatten(1); G = (G - G.mean(1, keepdim=True)) / G.std(1, keepdim=True).clamp(min=1e-6); best = (G @ Xt.T / Xt.shape[1]).max(1).values
print(f"\nI5a G2: nearest training image pixel corr: median {float(best.median()):.3f}, share > .8 {float((best > .8).float().mean()):.3f}, share > .6 {float((best > .6).float().mean()):.3f}")
# P2 proxy: delete conv2 channels 16-31; write f_del(p) ~ B f_keep(p) + beta, fit by lsq or in the output Fisher metric
keep, dele = torch.arange(24, device=DEV), torch.arange(24, 32, device=DEV)
Wfc = net.fc.weight.reshape(4, 32, 64); bfc = net.fc.bias
def logits_with(f): return torch.einsum("ocp,ncp->no", Wfc, f) + bfc
def fit(X, fisher, w=None):
    w = torch.ones(len(X), device=DEV) if w is None else w * len(X)
    f = feats(X).reshape(len(X), 32, 64); fk = torch.cat([f[:, keep], torch.ones_like(f[:, :1])], 1)   # (n, 25, 64)
    tgt = f[:, dele]                                                                                     # (n, 16, 64)
    if not fisher:
        A = torch.einsum("n,nkp,njp->kj", w, fk, fk); b = torch.einsum("n,nkp,ndp->kd", w, fk, tgt); return torch.linalg.lstsq(A.cpu(), b.cpu(), rcond=1e-7).solution.T.to(DEV)   # (8, 25)
    # output change = sum_d,p Wfc[:, del_d, p] (tgt - B fk)[d, p]; minimise sum_n e^T F_n e with F_n = diag p - p p^T
    p = F.softmax(net(X), 1); Fm = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]                    # (n, 4, 4)
    Wd = Wfc[:, dele]                                                                                    # (4, 16, 64)
    Phi = torch.einsum("odp,nkp->nodk", Wd, fk).reshape(len(X), 4, 8 * 25)                              # d(out)/d(B[d,k])
    y0 = torch.einsum("odp,ndp->no", Wd, tgt)
    A = torch.einsum("n,noi,nop,npj->ij", w, Phi, Fm, Phi); b = torch.einsum("n,noi,nop,np->i", w, Phi, Fm, y0)
    return torch.linalg.lstsq(A.cpu(), b.cpu()[:, None], rcond=1e-7).solution[:, 0].to(DEV).reshape(8, 25)   # min-norm: a zero metric gives B = 0
def score(B):
    f = feats(Xeval).reshape(5000, 32, 64); fk = torch.cat([f[:, keep], torch.ones_like(f[:, :1])], 1); g = f.clone()
    g[:, dele] = torch.einsum("dk,nkp->ndp", B, fk) if B is not None else 0
    lt = F.log_softmax(logits_with(f), 1); ls = F.log_softmax(logits_with(g), 1); return (lt.exp() * (lt - ls)).sum(1)
base = score(None); print(f"\nP2 proxy (delete conv2 channels 24-31), KL on 5000 fresh true samples: no repair {float(base.mean()):.4f}")
ref = {}
for k, X in sets.items():
    for fisher in [False, True]:
        kl = score(fit(X, fisher, W.get(k))); ref[(k, fisher)] = kl
for fisher in [False, True]:
    t = ref[("truth (fresh)", fisher)]
    for k in sets:
        v = ref[(k, fisher)]; d = v - t
        print(f"  {'Fisher' if fisher else 'lsq   '} write from {k:16s}: KL {float(v.mean()):.4f} (se {float(v.std() / 5000 ** .5):.4f}) | paired vs truth {float(d.mean()):+.4f} (se {float(d.std() / 5000 ** .5):.4f})")
