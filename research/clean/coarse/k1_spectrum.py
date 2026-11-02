"""K1 step 2a (dev only): do verified families share a small common core in the children's weights?

Pre-registered (plan, 2026-10-19): verified families need 30% fewer ranks than random groups of the same size to
reach 90% of the energy, per weight matrix (gate rows, value rows, output columns; each row scaled to unit length).
Also reported, not used for the decision: top-1 energy share, and the same on the children's ACTIVATIONS and OUTPUTS.

  python3 research/clean/coarse/k1_spectrum.py --layers 28,15,3
Writes results/clean-k1-spectrum.json.
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from r2lib import Setup, ROOT  # noqa: E402

TAGS = {28: "global", 15: "global", 3: "local_rand"}


def spec(M):
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    s = np.linalg.svd(M, compute_uv=False) ** 2
    e = np.cumsum(s) / s.sum()
    return int(np.searchsorted(e, 0.9) + 1), float(e[0])


def views(S, idx, sg):
    n = S.n_old
    G, U, D = S.G[idx], S.U[idx] * sg[:, None], (S.D[:, idx] * sg[None, :]).T
    H = S.Hh[:, idx] * sg[None, :]
    Hc = H - H.mean(0)                                    # activations over held-out positions, one row per child
    Y = np.stack([np.outer(H[:2000, i], D[i]).ravel() for i in range(len(idx))])  # each child's output on 2000 positions
    return {"gate": G, "value": U, "out": D, "act": Hc.T, "child_output": Y}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="28,15,3")
    ap.add_argument("--nulls", type=int, default=20)
    a = ap.parse_args()
    out = {"rule": "pass if mean ranks@90% (families) <= 0.7 x mean ranks@90% (random groups same size), per view",
           "rows": []}
    rng = np.random.default_rng(0)
    for L in map(int, a.layers.split(",")):
        S = Setup(L, 48, TAGS[L])
        acc = {}
        for f in S.families:
            n = len(f["idx"])
            for name, M in views(S, f["idx"], f["sg"]).items():
                r, t1 = spec(M)
                acc.setdefault(name, {"fam_rank": [], "fam_top1": [], "null_rank": [], "null_top1": [], "n": []})
                acc[name]["fam_rank"].append(r); acc[name]["fam_top1"].append(t1); acc[name]["n"].append(n)
                for _ in range(a.nulls):
                    ridx = rng.choice(S.n_old, n, replace=False)
                    rr, rt = spec(views(S, ridx, np.ones(n))[name]) if name in ("act", "child_output") else spec(
                        {"gate": S.G[ridx], "value": S.U[ridx], "out": S.D[:, ridx].T}[name])
                    acc[name]["null_rank"].append(rr); acc[name]["null_top1"].append(rt)
        for name, v in acc.items():
            fr, nr = np.mean(v["fam_rank"]), np.mean(v["null_rank"])
            row = {"layer": L, "tag": TAGS[L], "view": name, "families": len(v["fam_rank"]),
                   "mean_children": float(np.mean(v["n"])), "fam_rank90": float(fr), "null_rank90": float(nr),
                   "ratio": float(fr / nr), "pass": bool(fr <= 0.7 * nr),
                   "fam_top1": float(np.mean(v["fam_top1"])), "null_top1": float(np.mean(v["null_top1"]))}
            out["rows"].append(row)
            print(f"L{L:2d} {name:13s} fams {row['families']:2d} kids {row['mean_children']:.1f}  rank90 fam "
                  f"{fr:.2f} null {nr:.2f} ratio {row['ratio']:.2f} {'PASS' if row['pass'] else 'fail'}  "
                  f"top1 fam {row['fam_top1']:.2f} null {row['null_top1']:.2f}", flush=True)
        json.dump(out, open(os.path.join(ROOT, "results", "clean-k1-spectrum.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
