# scale: neurons=A:block0-all; net=A, C-ladder, W-ladder, L-ladder, R4-ladder, WC-ladder, R4C-ladder; photos=saved; where=mac
# rung: Q=plan "Read what training adds on top of the calculation", step 1: where training's write change lies; N=weights only; EST=1 min Mac; PERMIT=Shonu approved plan 2026-11-25
"""Step 1 (Mac, numpy, weights only). For every kept block-0 neuron j, its effective write column
V[:, j] = bn2 scale (per output) x conv2[:, j] (512 x 3 x 3 = 4608 numbers). A change of the writes between two builds
is split into three parts:
  Dspan  inside the span of A's deleted write columns (C adds only here, by construction)
  Kspan  inside the span of A's kept write columns, outside Dspan (includes the neuron's own write: rescaling)
  rest   outside every A write column (new directions)
Then a "trained beta": the Dspan part of (X - A) written as a mix of deleted writes, against C's beta.
Also the bn2 shift change (the constant term C does not have).
Output: results/oc_vs/writes.json and the console table."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAD = os.path.join(HERE, "..", "results", "oc_ladder")
R18 = os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18")
OUT = os.path.join(HERE, "..", "results", "oc_vs")
EPS = 1e-5
CUTS = (11, 64, 192, 384)
BUILDS = ("C", "W", "WC", "L", "R4", "R4C")


def bn_parts(bn):
    s = bn[0] / np.sqrt(bn[3] + EPS)
    return s, bn[1] - bn[2] * s


def writes(w):
    s2, t2 = bn_parts(w["bn2_0"])
    V = (w["W2_0"] * s2[:, None, None, None]).astype(np.float64)          # out x in x 3 x 3
    return V.transpose(1, 0, 2, 3).reshape(512, -1), t2.astype(np.float64)  # row j = write of neuron j (4608)


def basis(M):
    """Orthonormal basis of the row span of M (rows = vectors)."""
    q, r = np.linalg.qr(M.T)
    keep = np.abs(np.diag(r)) > 1e-10 * np.abs(np.diag(r)).max()
    return q[:, keep]


def split(Dl, QD, QK):
    """Shares of the total squared norm of the rows of Dl in Dspan, Kspan (outside Dspan), rest."""
    tot = (Dl ** 2).sum()
    pD = Dl @ QD @ QD.T
    r1 = Dl - pD
    pK = r1 @ QK @ QK.T
    return dict(norm=float(np.sqrt(tot)), Dspan=float((pD ** 2).sum() / tot), Kspan=float((pK ** 2).sum() / tot),
                rest=float(((r1 - pK) ** 2).sum() / tot))


def main():
    os.makedirs(OUT, exist_ok=True)
    VA, tA = writes(np.load(os.path.join(R18, "w_A.npz")))
    res = {}
    for n in CUTS:
        d = os.path.join(LAD, f"n{n}")
        jt = np.load(os.path.join(d, "jobtest_L.npz"))
        D, K = jt["D"], jt["K"]
        QD = basis(VA[D])
        QKo = basis(VA[K] - VA[K] @ QD @ QD.T)                               # kept writes, outside Dspan
        floor = dict(Dspan=QD.shape[1] / 4608, Kspan=QKo.shape[1] / 4608)
        beta_C = np.load(os.path.join(d, "beta_C.npy")).astype(np.float64)   # nD x nK
        V = {b: writes(np.load(os.path.join(d, f"w_{b}.npz"))) for b in BUILDS}
        V["A"] = (VA, tA)
        r = dict(floor=floor, nD=len(D), nK=len(K))
        # check: C - A equals the fold sum_d beta_dj V_A[d] (bn2 unchanged in C)
        fold = beta_C.T @ VA[D]
        r["check_C_fold_maxerr"] = float(np.abs((V["C"][0][K] - VA[K]) - fold).max() / np.abs(fold).max())
        for x, ref in (("C", "A"), ("W", "A"), ("WC", "A"), ("L", "A"), ("R4C", "A"), ("WC", "C"), ("R4C", "C"),
                       ("W", "C")):
            Dl = V[x][0][K] - V[ref][0][K]
            row = split(Dl, QD, QKo)
            # trained beta: least-squares mix of deleted writes for each kept neuron's change from A
            if ref == "A" and x != "C":
                bt = np.linalg.lstsq(VA[D].T, (V[x][0][K] - VA[K]).T, rcond=None)[0]      # nD x nK
                row["beta_corr_with_C"] = float(np.corrcoef(bt.ravel(), beta_C.ravel())[0, 1])
                row["beta_norm_ratio"] = float(np.linalg.norm(bt) / np.linalg.norm(beta_C))
                cj = (bt * beta_C).sum(0) / (np.linalg.norm(bt, axis=0) * np.linalg.norm(beta_C, axis=0) + 1e-30)
                row["beta_colcos_median"] = float(np.median(cj))
            dt = V[x][1] - V[ref][1]
            row["bn2_shift_change_norm"] = float(np.linalg.norm(dt))
            row["bn2_shift_A_norm"] = float(np.linalg.norm(tA))
            # summed over 3 x 3 (the vector vecread reads): share in the span of the summed deleted writes
            S = Dl.reshape(len(K), 512, 9).sum(2)
            QDs = basis(VA[D].reshape(len(D), 512, 9).sum(2))
            row["Dspan_sum"] = float(((S @ QDs) ** 2).sum() / (S ** 2).sum())
            row["Dspan_sum_floor"] = QDs.shape[1] / 512
            # own-write part: share of each neuron's change along its own A write
            own = ((Dl * VA[K]).sum(1) ** 2 / ((VA[K] ** 2).sum(1) + 1e-30)).sum() / (Dl ** 2).sum()
            row["own_share"] = float(own)
            r[f"{x}-{ref}"] = row
        res[str(n)] = r
    json.dump(res, open(os.path.join(OUT, "writes.json"), "w"), indent=1)

    print("Floors (random direction): Dspan = nD/4608, Kspan = nK/4608 (outside Dspan)")
    for n in CUTS:
        r = res[str(n)]
        print(f"\ncut {n}: floor Dspan {r['floor']['Dspan']:.3f}, Kspan {r['floor']['Kspan']:.3f}; "
              f"check C fold max rel err {r['check_C_fold_maxerr']:.1e}")
        print(f"  rest floor (random) {1 - r['floor']['Dspan'] - r['floor']['Kspan']:.3f}; "
              f"summed Dspan floor {r['C-A']['Dspan_sum_floor']:.3f}")
        print(f"  {'change':8s} {'norm':>7s} {'Dspan':>6s} {'Kspan':>6s} {'rest':>6s} {'Dsum':>6s} {'own':>6s} {'bn2dt':>6s}"
              f" {'beta corr':>9s} {'beta ratio':>10s} {'col cos':>7s}")
        for k, v in r.items():
            if "-" not in k:
                continue
            b = (f" {v['beta_corr_with_C']:9.3f} {v['beta_norm_ratio']:10.3f} {v['beta_colcos_median']:7.3f}"
                 if "beta_corr_with_C" in v else "")
            print(f"  {k:8s} {v['norm']:7.3f} {v['Dspan']:6.3f} {v['Kspan']:6.3f} {v['rest']:6.3f} {v['Dspan_sum']:6.3f} "
                  f"{v['own_share']:6.3f} {v['bn2_shift_change_norm']:6.3f}{b}")


if __name__ == "__main__":
    main()
