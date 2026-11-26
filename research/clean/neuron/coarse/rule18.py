"""N11 step 3 (numpy only): do coarse neurons built from A alone match B's neurons?

Candidates, all from A (no B input):
  R-parent(P)    = stage-3 channel P itself (read_s3 pos: max over positions)
  R-children(P)  = per photo, the max over P's m children of the child's z-scored pos (max over positions commutes with
                   max over children, so this is the max over positions of the strongest child). Children of P = the m
                   A block-0 neurons j with the largest feed0[j, P] (path_A: P's contribution to j at j's peak, mean
                   over j's own top 50 photos; all classes, no B).
  single A(j)    = one A block-0 neuron (baseline)
Match = overlap of top-N photo sets: |top_N(candidate) & top_N(b)| / N, N = 200, held + test photos.
For every block-0 neuron b of a B network: the best match within each candidate family, and the best match among the
other seed's neurons (how well B itself repeats).
Usage: python rule18.py <B tag> <other seed tag> [m]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "coarse", "r18")
N = 200


def topsets(V):
    idx = np.argpartition(-V, N, axis=0)[:N]          # N x k
    S = np.zeros(V.shape, bool)
    S[idx, np.arange(V.shape[1])[None, :]] = True
    return S


def overlap(SX, SY):
    return (SX.T.astype(np.float32) @ SY.astype(np.float32)) / N   # kx x ky


def z(V):
    return (V - V.mean(0)) / V.std(0).clip(1e-6)


def main():
    tag, other = sys.argv[1], sys.argv[2]
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    A = np.load(os.path.join(OUT, "read_A.npz"))
    S3 = np.load(os.path.join(OUT, "read_s3.npz"))
    assert (A["ix"] == S3["ix"]).all()
    a = z(A["pos0"].astype(np.float32))
    feed = np.load(os.path.join(OUT, "path_A.npz"))["feed0"]          # 512 x 256
    kids = np.argsort(-feed, axis=0)[:m].T                             # 256 x m
    Rch = np.stack([a[:, kids[P]].max(1) for P in range(256)], 1)    # photos x 256
    fam = {"single A neuron": topsets(a), "R-parent": topsets(S3["pos"].astype(np.float32)),
           f"R-children (m={m})": topsets(Rch)}
    B = np.load(os.path.join(OUT, f"read_{tag}.npz"))
    Bo = np.load(os.path.join(OUT, f"read_{other}.npz"))
    SB, SO = topsets(B["pos0"].astype(np.float32)), topsets(Bo["pos0"].astype(np.float32))
    res = {}
    best = {}
    for name, S in fam.items():
        O = overlap(S, SB)
        best[name] = (O.max(0), O.argmax(0))
    best["other seed"] = (overlap(SO, SB).max(0), overlap(SO, SB).argmax(0))
    k = SB.shape[1]
    print(f"{tag}: {k} block-0 neurons; match = top-{N} overlap; median / 75th pct / share of neurons where it beats single A")
    for name, (v, _) in best.items():
        beats = float((v > best["single A neuron"][0]).mean())
        print(f"  {name:22s} median {np.median(v):.2f}  75th {np.percentile(v, 75):.2f}  beats single A in {beats:.2f}")
        res[name] = dict(median=float(np.median(v)), p75=float(np.percentile(v, 75)), beats_single=beats, per_neuron=v.tolist())
    for j in (122, 105, 121):
        if j < k:
            row = ", ".join(f"{n} {best[n][0][j]:.2f} (#{best[n][1][j]})" for n in best)
            print(f"  B n{j}: {row}")
    v_rc, v_sa = best[f"R-children (m={m})"][0], best["single A neuron"][0]
    order = np.argsort(-(v_rc - v_sa))[:8]
    print("  largest gains of R-children over single A:", ", ".join(f"n{j} {v_rc[j]:.2f} vs {v_sa[j]:.2f} (P s3:{best[f'R-children (m={m})'][1][j]})" for j in order))
    json.dump(res, open(os.path.join(OUT, f"rule_{tag}_m{m}.json"), "w"))


if __name__ == "__main__":
    main()
