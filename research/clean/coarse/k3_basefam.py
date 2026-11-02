"""K3 (dev only): families from the BASE MODEL alone, and a gate-widening coarsener built on them.

Part A. Base families. Parent candidates = both poles of the cut's K kept neurons (base-only). A pole j is a child of
parent pole i if i's percentile at j's top-1% positions has median >= 0.95 (R1 threshold). Hub guard: a pole that is a
child of > 20% of parent poles does not count. Compared with the trained-parent families (dev only, for checking):
share of a trained family's children that the matched base parent also lists. Null: a random kept pole's children.

Part B. Widen. Each kept neuron gets a gate bias b = alpha * (median own pre-activation at its own hot positions
- median pre-activation at its children's hot positions), alpha in {0.5, 1} (fixed before running), then LS output
refit. Dev CE (test blocks 0-19) vs cut_ls.

  python3 research/clean/coarse/k3_basefam.py --layers 28,15
Writes results/clean-k3-basefam.json.
"""
import argparse, json, os, sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from contract import mlp_io, cut_ls, orig_weights  # noqa: E402
from r2lib import Setup, ROOT, HUB  # noqa: E402
from tree_lib import token_blocks, eval_loss, silu, ls_down  # noqa: E402


def ranks01(A):
    """Column-wise percentile ranks in [0, 1] (float32)."""
    R = np.empty(A.shape, dtype=np.float32)
    for c in range(A.shape[1]):
        R[np.argsort(A[:, c]), c] = np.linspace(0, 1, A.shape[0], dtype=np.float32)
    return R


def base_families(H, kept, q=0.01, thr=0.95):
    n = H.shape[1]
    P = np.concatenate([H, -H], 1)                       # pole p < n: +, p >= n: -
    nh = max(1, int(len(P) * q))
    hot = np.argpartition(-P, nh, axis=0)[:nh].T         # (2n, nh)
    parents = np.concatenate([kept, kept + n])
    Rp = ranks01(P[:, parents])
    cov = np.stack([np.median(Rp[:, k][hot], axis=1) for k in range(len(parents))])  # (2K, 2n)
    for k, p in enumerate(parents):                     # a parent's own neuron is not its child
        cov[k, p % n] = cov[k, p % n + n] = 0
    kids = cov >= thr
    hubs = kids.mean(0) > HUB
    kids &= ~hubs[None, :]
    return parents, [np.where(r)[0] for r in kids], hot


class BiasSwiGLU(nn.Module):
    def __init__(self, G, b, U, D):
        super().__init__()
        self.G, self.b, self.U, self.D = (mx.array(np.asarray(a, dtype=np.float32)) for a in (G, b, U, D))

    def units(self, x):
        return nn.silu(x @ self.G.T + self.b) * (x @ self.U.T)

    def __call__(self, x):
        return self.units(x) @ self.D.T

    def n_params(self):
        return int(self.G.size + self.b.size + self.U.size + self.D.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="28,15")
    ap.add_argument("--K", type=int, default=48)
    a = ap.parse_args()
    out = {"rows": []}
    rng = np.random.default_rng(0)
    for L in map(int, a.layers.split(",")):
        S = Setup(L, 48, "global")
        model, tok = S.model, S.tok
        dev = token_blocks(tok, "test", 60)[:20]
        X = S.Xc                                          # 64 cal blocks, first position dropped
        G, U, D = orig_weights(model, L)
        H = S.Hc
        Gk, Uk, Dk, kept = cut_ls(model, L, a.K, X)
        parents, kids, hot = base_families(H, kept)
        n = H.shape[1]
        # --- A: overlap with trained families (dev check only)
        ov, ovn = [], []
        Ak = H[:, kept]
        for f in S.families:
            c = np.corrcoef(np.c_[f["aT_cal"], Ak].T)[0, 1:]
            k = int(np.argmax(np.abs(c)))
            pk = k if c[k] > 0 else k + a.K
            tk = set(int(x) for x in f["kids"])
            ov.append(len(tk & set(kids[pk].tolist())) / len(tk))
            ovn.append(len(tk & set(kids[rng.integers(len(parents))].tolist())) / len(tk))
        sizes = [len(k) for k in kids]
        # --- B: widen + LS refit
        base_ce = eval_loss(model, dev)["ce"]
        orig = model.model.layers[L].mlp
        res = {}
        for alpha in (0.0, 0.5, 1.0):
            b = np.zeros(a.K)
            pre = X @ Gk.T
            for k in range(a.K):
                kp = [j for j in kids[k].tolist() + kids[k + a.K].tolist()]
                if not kp:
                    continue
                own_hot = np.argsort(-H[:, kept[k]])[: hot.shape[1]]
                kid_hot = np.unique(hot[kp].ravel())
                b[k] = alpha * (np.median(pre[own_hot, k]) - np.median(pre[kid_hot, k]))
            A = silu(X @ Gk.T + b) * (X @ Uk.T)
            Y = H @ D.T
            Dn = np.linalg.solve(A.T @ A + 1e-4 * np.trace(A.T @ A) / a.K * np.eye(a.K), A.T @ Y).T
            model.model.layers[L].mlp = BiasSwiGLU(Gk, b, Uk, Dn)
            res[alpha] = eval_loss(model, dev)["ce"] - base_ce
            model.model.layers[L].mlp = orig
        row = {"layer": L, "K": a.K, "parent_poles_with_2plus_kids": float(np.mean(np.array(sizes) >= 2)),
               "median_kids": float(np.median(sizes)), "trained_family_overlap": float(np.mean(ov)),
               "overlap_null": float(np.mean(ovn)), "dce_widen": {str(k): float(v) for k, v in res.items()}}
        out["rows"].append(row)
        print(row, flush=True)
        json.dump(out, open(os.path.join(ROOT, "results", "clean-k3-basefam.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
