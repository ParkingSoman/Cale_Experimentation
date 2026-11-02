"""K5 (Part VIII): sealed split, one look, for the same-image steps of this session."""
import numpy as np, torch, time
SRC = open("c47_netprec.py").read(); exec(SRC.split("XQ = np.load")[0])
C27 = np.load("cache/c27_cold.npz"); RS = {"kbar(S)": C27["kbar"], "T": C27["T"]}
def make(X, gc, P):
    d = -P(gc); c = 0.0
    for i in range(0, len(X), 8):
        torch.manual_seed(i); c += float(((gam(X[i:i + 8]) * d).sum((2, 3)) ** 2).sum())
    return build(Vs["S"] * mask + (-float((gc * d).sum()) / (c / len(X))) * d)
Xr = D.X[np.sort(np.random.default_rng(26).choice(D.tr, 256, replace=False))]
gr = push(Xr, Xr)                                                        # c26 recipe: full push, photo M
XQ = np.load("cache/c45_readerq_368.npy"); Xn = np.load("cache/e1_self_256.npy")
N = torch.load("cache/c47_netprec.pt", weights_only=False); Gn, An = N["Gn"].to(DEV), N["An"].to(DEV)
Gni = torch.linalg.inv(Gn + 1e-6 * Gn.trace() / NRD * torch.eye(NRD, device=DEV)); Ani = torch.linalg.inv(An + 1e-6 * An.trace() / 962 * torch.eye(962, device=DEV))
Pnet = lambda R: (Gni @ (R * free) @ Ani) * free
gcC = torch.load("cache/c41_split_rand.pt", weights_only=False)["class256"][2].to(DEV)
gcQ = torch.load("cache/c46_readerq368_grads.pt", weights_only=False)["gc"].to(DEV)
B = {"photo256_step": (Xr, gr, Pinv), "class256_cov": (Xn, gcC, Pinv), "readerq368_photoM": (XQ, gcQ, Pinv), "readerq368_netM": (XQ, gcQ, Pnet)}
R = {}
for k, (X, g, P) in B.items():
    r = per_image(make(X, g, P), D.se); R[k] = r
    print(f"{k:18s} sealed {r.mean():.4f} (se {r.std() / np.sqrt(len(r)):.4f})", flush=True); paired(k, r, RS)
np.savez("cache/c48_cold.npz", **R); print(f"done ({time.time() - t0:.0f}s)")
