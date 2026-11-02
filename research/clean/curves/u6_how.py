"""BEAT H: the HOW (PAPER section 14). For each candidate WHAT (Lam, 3 x 10), each unit's read = gated least squares
(Lemma 14.1) of t_j = sum_i Lam_ji relu(z_i) on S_j = {t_j > 0}, from the generalist mix (train images). Then:
realized second-order share (Theorem 10b with the realized pooled units), false-on rate, and dev-5000 KL under the
lsq simple-rule write (the same write that gave M's trained read .0798). Predictions in CURVES-LOG.md (BEAT, H)."""
import numpy as np, torch
from inception import *
from family import Data, Family, pool3
from w_fisher import load_build, real_kl
from u4_w3 import lsq_write
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
W5 = m.mixed3b_5x5_pre_relu_conv.weight.detach()[[i - 320 for i in fam]]; b5 = m.mixed3b_5x5_pre_relu_conv.bias.detach()[[i - 320 for i in fam]]
u = np.load("cache/u5_what.npz"); eye = np.eye(10)
cands = {"kept 10/120/250 (E2 check)": eye[[0, 3, 6]], "bumps 1+cos": u["bump"], "M start cells": u["M_L"]}
for r in range(4): cands[f"arcs #{r + 1}"] = u["top_L"][r]
names = list(cands); L = torch.tensor(np.concatenate([cands[n] for n in names]), dtype=torch.float32, device=DEV)  # (3C, 10)
nu = L.shape[0]; P = 801
G = torch.zeros(nu, P, P, dtype=torch.float64); r = torch.zeros(nu, P, dtype=torch.float64)
rng = np.random.default_rng(21); fit = np.sort(rng.choice(D.tr, 2000, replace=False))
rest = np.setdiff1d(D.tr, fit); held = np.sort(rng.choice(rest, 500, replace=False)); wimg = np.sort(rng.choice(np.setdiff1d(rest, held), 1000, replace=False))
def windows(ins):
    Xu = F.unfold(F.pad(ins[2], (2, 2, 2, 2)), 5)                                   # (B, 800, 784)
    return torch.cat([Xu, torch.ones_like(Xu[:, :1])], 1).transpose(1, 2).reshape(-1, P)  # (B*784, 801)
with torch.no_grad():
    for i in range(0, len(fit), 20):
        ins = inputs_3b(m, to_3a(m, prep(D.X[fit[i:i + 20]]))); X = windows(ins)
        z = pre_3b(m, ins[0])[:, fam].permute(0, 2, 3, 1).reshape(-1, 10)
        if i == 0:   # instrument: the unfolded windows reproduce the members' pre-activations
            zz = X[:, :800] @ W5.reshape(10, -1).T + b5; print("window check max |err| / sd:", float((zz - z).abs().max() / z.std()), flush=True)
        T = F.relu(z) @ L.T                                                             # (N, 3C) targets
        for j in range(nu):
            on = T[:, j] > 0; Xs = X[on]
            G[j] += (Xs.T @ Xs).cpu().double(); r[j] += (Xs.T @ T[on, j]).cpu().double()
        if (i // 20) % 20 == 0: print(f"  fit {i + 20}/{len(fit)}", flush=True)
V = torch.stack([torch.linalg.solve(G[j] + 1e-7 * torch.trace(G[j]) / P * torch.eye(P, dtype=torch.float64), r[j]) for j in range(nu)]).float().to(DEV)
d4 = np.load("cache/t4.npz"); Phi = np.load("cache/t2.npz")["Phi"]; tot = np.trace(Phi @ (d4["Szz"] + np.outer(d4["mu"], d4["mu"])))
builds = {"M trained read": load_build(m, fam, "L-k3-m-lr3e-2")}
for ci, n in enumerate(names):
    fm = Family(m, fam, 3).to(DEV); v = V[3 * ci:3 * ci + 3]
    fm.set_weights(v[:, :800].reshape(3, 32, 5, 5), v[:, 800], [c[:, :3] for c in builds["M trained read"].weights()[2]])
    builds[n] = fm
    if n.startswith("kept"):
        cs = F.cosine_similarity(v[:, :800], W5.reshape(10, -1)[[0, 3, 6]], 1); print("E2 check: kernel cos to members", cs.cpu().numpy(), "bias", v[:, 800].cpu().numpy(), "vs", b5[[0, 3, 6]].cpu().numpy())
# realized second-order share and false-on rate on held-out train images
stat = {n: dict(Z=[], C=[], fo=[], on=[]) for n in builds}
with torch.no_grad():
    for i in range(0, len(held), 50):
        ins = inputs_3b(m, to_3a(m, prep(D.X[held[i:i + 50]]))); zr = F.relu(pre_3b(m, ins[0])[:, fam])
        zp = pool3(zr)[:, :, 2:12, 2:12].permute(1, 0, 2, 3).reshape(10, -1)
        for ci, (n, fm) in enumerate(builds.items()):
            c = F.relu(fm.pre(ins)); cp = pool3(c)[:, :, 2:12, 2:12].permute(1, 0, 2, 3).reshape(3, -1)
            stat[n]["Z"].append(zp.cpu().double().numpy()); stat[n]["C"].append(cp.cpu().double().numpy())
            if n in cands:
                t = torch.einsum("jk,bkhw->bjhw", torch.tensor(cands[n], dtype=torch.float32, device=DEV), zr)
                stat[n]["fo"].append(float(((t <= 0) & (c > 0)).float().sum() / (t <= 0).float().sum().clamp(min=1)))
            stat[n]["on"].append(float((c > 0).float().mean()))
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print("\nrealized second-order share (beta .260 on top) | false-on | on-rate")
for n in builds:
    Z = np.concatenate(stat[n]["Z"], 1); C = np.concatenate(stat[n]["C"], 1); S = np.cov(np.concatenate([Z, C]), bias=True)
    Szc, Scc = S[:10, 10:], S[10:, 10:]
    sh = np.trace(np.linalg.solve(Scc + 1e-9 * np.eye(3), Szc.T @ Phi @ Szc)) / tot
    print(f"  {n:28s} {sh:.3f} | {np.mean(stat[n]['fo']) if stat[n]['fo'] else float('nan'):.3f} | {np.mean(stat[n]['on']):.3f}", flush=True)
# functional: dev-5000 KL under the lsq simple-rule write (1000 images), paired against M's read under the same write
B = {n: (fm, *lsq_write(m, fam, D, fm, wimg, 3)) for n, fm in builds.items()}
K = real_kl(m, fam, D, D.dv, B)
for n, v in K.items():
    d = v - K["M trained read"]; print(f"  {n:28s} lsq write: dev-5000 KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f}) | paired vs M read {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})", flush=True)
torch.save(dict(V=V.cpu(), names=names, cands=cands), "cache/u6_how.pt"); np.savez("cache/u6_how_kl.npz", **{n.split()[0] + str(i): v for i, (n, v) in enumerate(K.items())})
