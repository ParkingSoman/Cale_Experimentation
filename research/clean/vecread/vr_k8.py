"""K8 (added 2026-11-23 after the write geometry disagreed with its own data side; rule fixed before more n101
readings): for real writes, the top-5 geometry entries agree with the top-5 data-side entries (examples, IoU).
Test vectors (inputs only): all writes of network A into the space, 3 x 3 summed, bn-scaled:
  s3   stage-3 block-1 writes (layer3[1].conv2, 256)
  s4b0 stage-4 block-0 writes (W2_0, 512)
  s4b1 stage-4 block-1 writes (W2_1, 512)
Score = mean top-5 overlap. Null = the same with geometry of write i paired with the data side of write j (1000
shuffles). Pass = above the null's 99th percentile. Write metric = the best of raw / whitened / shrunk on held set 1;
`--confirm` reruns the chosen metric once on held set 2.
Usage: python vr_k8.py [--confirm]
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

import vecread as vr
from cards_n101 import R18, bn_scale

RNG = np.random.default_rng(8)


def writes(space):
    if space == "s3":
        import torchvision
        m = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        b = m.layer3[1]
        W = b.conv2.weight.detach().numpy()
        bn = b.bn2
        sc = (bn.weight / (bn.running_var + bn.eps).sqrt()).detach().numpy()
        return (W * sc[:, None, None, None]).sum((2, 3)).T
    A = np.load(os.path.join(R18, "w_A.npz"))
    k = "0" if space == "s4b0" else "1"
    return (A[f"W2_{k}"] * bn_scale(A[f"bn2_{k}"])[:, None, None, None]).sum((2, 3)).T


def data_top5(held, dic, V):
    """Top-5 entries by IoU on each write's top-1% held cells."""
    vals = held.project(dic.space, V)                         # n, cells, k
    n, cells, k = vals.shape
    KK = np.nonzero(dic.kept)[0]
    rows, cols = [], []
    for r, e in enumerate(KK):
        p, ce = held.cells(dic, e)
        rows.append(np.full(len(p), r))
        cols.append(p * cells + ce)
    M = sp.csr_matrix((np.ones(sum(map(len, rows))), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(KK), n * cells))
    size = np.asarray(M.sum(1)).ravel()
    X = vals.reshape(n * cells, k)
    hot = (X >= np.quantile(X, 0.99, axis=0)[None]).astype(np.float32)
    inter = M @ hot                                           # entries x writes
    iou = inter / (size[:, None] + hot.sum(0)[None] - inter)
    return np.argsort(-iou, axis=0)[:5].T                     # writes x 5 (indices into kept)


def geom_top5(dic, V, metric):
    D = dic.unit(metric)
    U = dic.metric(V, metric)
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    return np.argsort(-(U @ D.T), axis=1)[:, :5]


def score(G, E):
    ov = np.array([len(set(g) & set(e)) for g, e in zip(G, E)])
    null = [np.mean([len(set(g) & set(e)) for g, e in zip(G, E[RNG.permutation(len(E))])]) for _ in range(1000)]
    return float(ov.mean()), float(np.percentile(null, 99))


def main():
    confirm = "--confirm" in sys.argv
    out_p = os.path.join(vr.RES, "vr_k8.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    held = vr.Held("_h2" if confirm else "")
    for space in ("s4b0", "s4b1", "s3"):
        dic = vr.Dictionary(space)
        V = writes(space)
        E = data_top5(held, dic, V)
        if not confirm:
            r = {}
            for metric in ("raw", "whitened", "shrunk"):
                m, null99 = score(geom_top5(dic, V, metric), E)
                r[metric] = {"mean_overlap": round(m, 3), "null_99th": round(null99, 3), "pass": bool(m > null99)}
            best = max(("raw", "whitened", "shrunk"), key=lambda k: r[k]["mean_overlap"])
            r["write_metric"] = best
            r["n_writes"] = len(V)
            res[space] = r
        else:
            best = res[space]["write_metric"]
            m, null99 = score(geom_top5(dic, V, best), E)
            res[space]["confirm"] = {"metric": best, "mean_overlap": round(m, 3), "null_99th": round(null99, 3),
                                     "pass": bool(m > null99)}
        print(space, json.dumps(res[space]), flush=True)
        json.dump(res, open(out_p, "w"), indent=1)


if __name__ == "__main__":
    main()
