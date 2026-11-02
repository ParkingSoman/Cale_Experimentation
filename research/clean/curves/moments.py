"""Compile stage: averages over a declared mix (weights + images in, sums out). Every output is a SUM with its total
weight, so mixes add: M(pi_1 Q_1 + pi_2 Q_2) is pi-weighted sums of the stored parts (Lemma 8.2 in PAPER.md).
Read part: X~ = 32 x ks x ks window of the 5x5 bottleneck (+1), H = moved sum of the family (C1 shifts, weights only).
usage: imported."""
import numpy as np, torch
from inception import *

LO, HI = 6, 22          # interior positions (no border effects), as in derive2.py


def c1_shifts():
    """Integer C1 shifts (weights only, card C1), in s1_fine family order; ks = window that holds the moved sum."""
    s = np.load("cache/c1_shifts.npz")["s"]; s = np.round(s - np.round(s.mean(0))).astype(int)
    return s, 5 + 2 * int(np.abs(s).max())


def moved_sum(z, s):
    """z (B,n,28,28) post-ReLU members -> H (B,28,28) = sum_i z_i(q + s_i), zero outside the map."""
    H = torch.zeros_like(z[:, 0])
    for i, (dy, dx) in enumerate(s):
        H += torch.roll(F.pad(z[:, i], (8, 8, 8, 8)), shifts=(-int(dy), -int(dx)), dims=(1, 2))[:, 8:36, 8:36]
    return H


def compile_read(m, X, fam, idx, w=None, flip=False, bs=20, seed=0):
    """Sums over images idx (weights w, default 1) of: G = X~ X~^T, c = X~ H, hh = H^2, family rate/mean/second moment.
    flip=True also adds each image's mirror (a symmetry of natural images)."""
    s, ks = c1_shifts(); p = ks // 2; d = 32 * ks * ks + 1
    w = np.ones(len(idx)) if w is None else np.asarray(w, float)
    G = torch.zeros(d, d, dtype=torch.float64, device="cpu"); c = torch.zeros(d, dtype=torch.float64); hh = 0.0
    on = torch.zeros(len(fam), dtype=torch.float64); mu = torch.zeros(len(fam), dtype=torch.float64)
    S2 = torch.zeros(len(fam), len(fam), dtype=torch.float64); W = 0.0; npos = (HI - LO) ** 2
    order = np.argsort(idx); idx = np.asarray(idx)[order]; w = w[order]
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            xb = np.asarray(X[idx[i:i + bs]]); wb = torch.tensor(w[i:i + bs], device=DEV, dtype=torch.float32)
            views = [xb, xb[:, :, ::-1]] if flip else [xb]
            for v in views:
                ins = inputs_3b(m, to_3a(m, prep(v))); z = F.relu(pre_3b(m, ins[0])[:, fam])
                H = moved_sum(z, s)[:, LO:HI, LO:HI]
                U = F.unfold(F.pad(ins[2], (p, p, p, p)), ks).reshape(len(v), d - 1, 28, 28)[:, :, LO:HI, LO:HI]
                U = torch.cat([U, torch.ones_like(U[:, :1])], 1)                       # (B, d, h, w)
                sw = wb.sqrt()[:, None, None, None]; Uw = (U * sw).permute(0, 2, 3, 1).reshape(-1, d)
                G += (Uw.T @ Uw).cpu().double()
                c += torch.einsum("bdhw,bhw,b->d", U, H, wb).cpu().double(); hh += float(torch.einsum("bhw,b->", H * H, wb))
                zz = z[:, :, LO:HI, LO:HI]
                on += torch.einsum("bihw,b->i", (zz > 0).float(), wb).cpu().double() / npos
                mu += torch.einsum("bihw,b->i", zz, wb).cpu().double() / npos
                S2 += torch.einsum("bihw,bjhw,b->ij", zz, zz, wb).cpu().double() / npos
                W += float(wb.sum())
    return dict(G=G, c=c, hh=hh, on=on, mu=mu, S2=S2, W=W, ks=ks, flip=flip)


def add(parts, pis):
    """Mix: sum_a pi_a M_a / W_a (each part normalised to its own mean first, then weighted by pi)."""
    out = {}
    for k in ["G", "c", "hh", "on", "mu", "S2"]:
        out[k] = sum(pi * P[k] / P["W"] for P, pi in zip(parts, pis))
    out["W"] = 1.0; out["ks"] = parts[0]["ks"]
    return out


def read_from(M, ridge=1e-6):
    """Least-squares read (MIX-DERIVED): K, b = argmin E_Q (H - K.X - b)^2. Ridge for numerics only."""
    G = M["G"] / M["W"]; c = M["c"] / M["W"]; d = G.shape[0]
    w = torch.linalg.solve(G + ridge * torch.trace(G) / d * torch.eye(d, dtype=G.dtype), c)
    ks = M["ks"]; return w[:-1].reshape(32, ks, ks).float(), float(w[-1])
