"""L2: single-layer coarsening arms on Pythia-160M at K units. Groups for merge arms: lineage (firing-corr clusters
at checkpoint t*), Ward (final input weights), random. Pre-registered rules: LINEAGE-LOG.md L2.

  python3 l2.py L K [tstar]
"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from common import *
from probe import probe

D = os.path.join(HF, "lineage-data")


class GMLP(nn.Module):
    def __init__(self, Wi, bi, Wo, bo):
        super().__init__()
        self.Wi, self.bi, self.Wo, self.bo = (mx.array(a.astype(np.float32)) for a in (Wi, bi, Wo, bo))

    def __call__(self, x):
        return nn.gelu(x @ self.Wi.T + self.bi) @ self.Wo.T + self.bo


def ls_out(Hc, Y, ridge=1e-4):
    A = np.c_[Hc, np.ones(len(Hc))]
    G = A.T @ A; G += np.eye(len(G)) * ridge * np.trace(G) / len(G)
    S = np.linalg.solve(G, A.T @ Y)
    return S[:-1].T, S[-1]


def corr_cols(A):
    A = A - A.mean(0); s = A.std(0); s[s < 1e-12] = np.inf; Z = A / s
    return (Z.T @ Z) / len(A)


def clusters(Cm, K, method="average"):
    Dm = np.clip(1 - np.abs(Cm), 0, 2); np.fill_diagonal(Dm, 0)
    g = fcluster(linkage(squareform(Dm, checks=False), method), K, "maxclust")
    return g - 1


def merge(Wi, bi, Wo, groups, X=None, Y=None, refit=False, rep=None):
    """One parent per group: input = mean of members (or the representative's input if rep given),
    output = sum of members' outputs (no fit) or LS refit."""
    ks = np.unique(groups)
    if rep is not None:
        Wi2 = Wi[rep]; bi2 = bi[rep]
    else:
        Wi2 = np.stack([Wi[groups == k].mean(0) for k in ks]); bi2 = np.array([bi[groups == k].mean() for k in ks])
    Wo2 = np.stack([Wo[:, groups == k].sum(1) for k in ks], 1)
    bo2 = None
    if refit:
        Wo2, bo2 = ls_out(gelu(X @ Wi2.T + bi2), Y)
    return Wi2, bi2, Wo2, bo2


def main():
    L, K = int(sys.argv[1]), int(sys.argv[2]); assert L in DEV_LAYERS
    tstar = sys.argv[3] if len(sys.argv) > 3 else json.load(open(os.path.join(RES, f"lineage-l1-L{L}.json")))["A"]["tstar"]
    model, tok = load_pythia()
    cal = mx.concatenate([blocks(tok, "train", 32), blocks(tok, "code_train", 32, skip=60)], 0)
    dev, code = blocks(tok, "test", 40), blocks(tok, "code_train", 40)
    Wi, bi, Wo, bo = (a.astype(np.float64) for a in mlp_w(model, L))
    X = mlp_in(model, cal, L).astype(np.float64)
    Hf = gelu(X @ Wi.T + bi); Y = Hf @ Wo.T + bo
    base = {"prose": block_ce(model, dev), "code": block_ce(model, code)}
    orig = layers(model)[L].mlp
    # group sources
    Zt = gelu(np.load(os.path.join(D, f"{tstar}.npz"))[f"Z{L}"].astype(np.float64))
    groups = {"lineage": clusters(corr_cols(Zt), K),
              "ward": fcluster(linkage(Wi / np.linalg.norm(Wi, axis=1, keepdims=True), "ward"), K, "maxclust") - 1,
              "random": np.random.default_rng(0).permutation(np.arange(3072) % K)}
    energy = (Hf ** 2).mean(0) * (Wo ** 2).sum(0)
    arms = {}
    top = np.sort(np.argsort(-energy)[:K]); Wo2, bo2 = ls_out(Hf[:, top], Y)
    arms["cut_ls"] = (Wi[top], bi[top], Wo2, bo2)
    for gname, g in groups.items():
        sizes = np.bincount(g)
        arms[f"{gname}_merge_nofit"] = merge(Wi, bi, Wo, g)[:3] + (bo,)
        arms[f"{gname}_merge_ls"] = merge(Wi, bi, Wo, g, X, Y, refit=True)
        rep = np.array([np.where(g == k)[0][np.argmax(energy[g == k])] for k in np.unique(g)])
        arms[f"{gname}_rep_ls"] = merge(Wi, bi, Wo, g, X, Y, refit=True, rep=rep)
        print(gname, "groups", len(sizes), "max size", sizes.max(), "singletons", int((sizes == 1).sum()), flush=True)
    out = {"layer": L, "K": K, "tstar": tstar, "base": {k: float(v.mean()) for k, v in base.items()}, "arms": {}}
    for name, (a, b, c, d) in arms.items():
        assert a.shape[0] == K
        layers(model)[L].mlp = GMLP(a, b, c, d if d is not None else bo)
        per = {t: block_ce(model, blk) - base[t] for t, blk in (("prose", dev), ("code", code))}
        r = {t: boot(v) for t, v in per.items()}
        r["per_block"] = {t: v.tolist() for t, v in per.items()}
        r["probe"] = probe(model, tok)["recite"]
        out["arms"][name] = r
        layers(model)[L].mlp = orig
        print(name, "prose %.4f [%.4f %.4f]" % tuple(r["prose"]), "code %.4f [%.4f %.4f]" % tuple(r["code"]), "recite %.3f" % r["probe"], flush=True)
    pb = lambda a, t: np.array(out["arms"][a]["per_block"][t])
    out["vs_cut"] = {a: {t: boot(pb(a, t) - pb("cut_ls", t)) for t in ("prose", "code")} for a in out["arms"]}
    out["lineage_vs_random"] = {a: {t: boot(pb(a, t) - pb(a.replace("lineage", "random"), t)) for t in ("prose", "code")}
                                for a in out["arms"] if a.startswith("lineage")}
    for a, v in out["vs_cut"].items():
        print("vs cut", a, "prose %.4f [%.4f %.4f]" % tuple(v["prose"]), "code %.4f [%.4f %.4f]" % tuple(v["code"]))
    save(f"l2-L{L}-K{K}", out)


if __name__ == "__main__":
    main()
