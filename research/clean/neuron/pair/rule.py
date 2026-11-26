"""Pair study: can a mechanical edit of the parent's read vector give the Small neuron's read direction?

Data: stream_cov.npz and stream_V.npy (stream.py). v_s = best linear read of Small neuron s from Base's u.
All comparisons use the covariance metric of u: cos_M(a, b) = a'Cb / sqrt(a'Ca b'Cb) (what the two vectors read).

Rules for the coarse read, from Base weights and Base data only:
  p          W1[p]
  pca_k      W1[p] projected on the top-k principal directions of u   (k = 768: Small's stream width; also 256, 512)
  cov        C W1[p] (the read direction pulled toward high-variance directions; a ridge-like smoothing)
  sib_k      W1[p] + the mean of the k Base read rows closest to it in the metric (siblings)
Score: median cos_M(rule, v_s) over the confident pairs, against cos_M(W1[p], v_s).
"""
import json
import os

import numpy as np

from cache import OUT


def main():
    d = np.load(os.path.join(OUT, "stream_cov.npz"))
    C, W1, agree, pmap = d["Cuu"], d["W1"], d["agree"], d["pmap"]
    V = np.load(os.path.join(OUT, "stream_V.npy")).astype(np.float64)
    ev, E = np.linalg.eigh(C)
    E = E[:, ::-1]
    cm = lambda a, b: float(a @ C @ b / np.sqrt((a @ C @ a) * (b @ C @ b)))
    Wm = W1 @ np.linalg.cholesky(C + 1e-6 * np.eye(len(C)))
    Wn = Wm / np.linalg.norm(Wm, axis=1, keepdims=True)
    out = {k: [] for k in ("p", "pca_256", "pca_512", "pca_768", "cov", "sib_5", "sib_20")}
    rng = np.random.default_rng(0)
    for s in rng.permutation(agree)[:800]:
        v, w = V[s], W1[pmap[s]]
        out["p"].append(cm(w, v))
        for k in (256, 512, 768):
            Ek = E[:, :k]
            out[f"pca_{k}"].append(cm(Ek @ (Ek.T @ w), v))
        out["cov"].append(cm(C @ w, v))
        sims = Wn @ Wn[pmap[s]]
        sims[pmap[s]] = -np.inf
        for k in (5, 20):
            nb = np.argsort(-sims)[:k]
            out[f"sib_{k}"].append(cm(w + W1[nb].mean(0) * np.linalg.norm(w) / np.linalg.norm(W1[nb].mean(0)), v))
    res = {k: float(np.median(x)) for k, x in out.items()}
    res["beats_p_share"] = {k: float(np.mean(np.array(x) > np.array(out["p"]))) for k, x in out.items() if k != "p"}
    res["variance_share_top768"] = float(ev[::-1][:768].sum() / ev.sum())
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, "rule.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
