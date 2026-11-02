"""Derived coarse neuron, step 2: the projection of the true quantity onto one neuron.
True quantity (S2/S3): H_r(q) = sum_theta relu(z_theta(q - r u_theta)), the fine family moved to its centres of
curvature (group average over rotations about q). One neuron = relu(K . x_q + b). K, b = the best linear read of H_r
from the neuron's input window x_q (ks x ks x 32 of the 5x5 bottleneck), from image statistics of the large model
(train images). No loss, no knob (ridge 1e-6 x mean diagonal, for numerical stability only). Write = Bayes write.
usage: derive2.py r ks [--eval dev|sealed] [--n 600]"""
import argparse, json, numpy as np, torch
from inception import *
from family import Family, Data, evaluate
from derive import bayes_write, PREF


def moved_sum_t(z, r):
    """z (B,10,28,28) post-ReLU members -> H (B,28,28): sum of members read at q - r u_theta (zero outside)."""
    H = torch.zeros_like(z[:, 0])
    for i, t in enumerate(PREF):
        dy, dx = int(round(r * np.sin(t))), int(round(-r * np.cos(t)))   # read at q + (dy, dx)
        H += torch.roll(F.pad(z[:, i], (8, 8, 8, 8)), shifts=(-dy, -dx), dims=(1, 2))[:, 8:36, 8:36]
    return H


def project(m, fam, D, r, ks, n=600, seed=21, lo=6, hi=22):
    idx = np.sort(np.random.default_rng(seed).choice(D.tr, n, replace=False)); p = ks // 2; d = 32 * ks * ks
    XtX = np.zeros((d + 1, d + 1)); Xty = np.zeros(d + 1); yy = 0.0; N = 0
    with torch.no_grad():
        for i in range(0, n, 20):
            ins = inputs_3b(m, to_3a(m, prep(D.X[idx[i:i + 20]])))
            H = moved_sum_t(F.relu(pre_3b(m, ins[0])[:, fam]), r)[:, lo:hi, lo:hi].reshape(-1)
            U = F.unfold(F.pad(ins[2], (p, p, p, p)), ks).reshape(-1, d, 28, 28)[:, :, lo:hi, lo:hi]
            U = U.permute(0, 2, 3, 1).reshape(-1, d); U = torch.cat([U, torch.ones_like(U[:, :1])], 1)
            XtX += (U.T @ U).cpu().numpy().astype(np.float64); Xty += (U.T @ H).cpu().numpy().astype(np.float64); yy += float((H * H).sum()); N += len(H)
    lam = 1e-6 * np.trace(XtX) / (d + 1)
    w = np.linalg.solve(XtX + lam * np.eye(d + 1), Xty)
    r2 = 1 - (yy - 2 * w @ Xty + w @ XtX @ w) / (yy - Xty[-1] ** 2 / N)
    return w[:-1].reshape(32, ks, ks).astype(np.float32), float(w[-1]), float(r2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("r", type=float); ap.add_argument("ks", type=int)
    ap.add_argument("--eval", default="dev"); ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    m = load_model(); D = Data(seed=0, ndev=1000 if a.eval == "dev" else 0); fam = list(np.load("cache/s1_fine.npz")["fam"])
    K, b, r2 = project(m, fam, D, a.r, a.ks, n=a.n)
    fm = Family(m, fam, 1, ks=a.ks).to(DEV); _, _, c0 = fm.weights()
    fm.set_weights(torch.tensor(K)[None].to(DEV), torch.tensor([b]).to(DEV), [c.to(DEV) for c in c0])
    cols, cbias, beta, alpha = bayes_write(m, fm, fam, D)
    fm.set_weights(torch.tensor(K)[None].to(DEV), torch.tensor([b]).to(DEV), [c.to(DEV) for c in cols]); fm.colbias = cbias
    kl_, se_, top = evaluate(m, fm, D, D.dv if a.eval == "dev" else D.se)
    with torch.no_grad():
        rate = (fm.pre(inputs_3b(m, to_3a(m, prep(D.X[np.sort(D.tr[:200])])))) > 0).float().mean().item()
    np.set_printoptions(precision=3, suppress=True)
    print(f"derived2 r={a.r} ks={a.ks}: projection R^2 {r2:.3f} | {a.eval} KL {kl_:.4f} (se {se_:.4f}) top-1 {top:.4f} | rate {rate:.3f} | beta {beta}", flush=True)
    tag = f"derived2-r{a.r:g}-ks{a.ks}"
    torch.save(dict(state=fm.state_dict(), colbias=[c.cpu() for c in cbias], fam=[int(f) for f in fam], k=1, ks=a.ks, r=float(a.r)), f"cache/ckpt/{tag}.pt")
    json.dump(dict(r=a.r, ks=a.ks, split=a.eval, r2=r2, kl=kl_, se=se_, top1=top, rate=rate, beta=beta.tolist()),
              open(f"../../../results/curves-{tag}-{a.eval}.json", "w"), indent=1)
