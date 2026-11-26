# scale: neurons=s3:69, A:children; net=A; photos=saved; where=mac
"""Read oc_other.npz on the Mac (LOG "Does coarsening hurt other capabilities?"). numpy only.

Job units: output units ranked by the DEV sum of the children's effect squared (centred logits, base - delete). Job set
J = the fewest units that hold 80% (also 50%, 90%). Picked from DEV only; every bar is read on the test photos.
KL split (exact chain rule): KL(P||Q) = KL over {J, not J} mass + P(J) KL inside J + P(not J) KL inside not J.
Usage: python3 oc_other_read.py [path to oc_other.npz]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
BUILDS = ["delete", "morph", "delmean", "morphmean"]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def kl_split(lb, ly, J):
    """Per photo: total KL, mass term, inside-J term, inside-not-J term (all >= 0, sum = total)."""
    P, Q = softmax(lb), softmax(ly)
    tot = (P * (np.log(P + 1e-30) - np.log(Q + 1e-30))).sum(1)
    pj, qj = P[:, J].sum(1), Q[:, J].sum(1)
    mass = pj * np.log(pj / qj) + (1 - pj) * np.log((1 - pj) / (1 - qj))

    def inside(idx, pm, qm):
        p, q = P[:, idx] / pm[:, None], Q[:, idx] / qm[:, None]
        return pm * (p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum(1)
    nJ = np.setdiff1d(np.arange(lb.shape[1]), J)
    return tot, mass, inside(J, pj, qj), inside(nJ, 1 - pj, 1 - qj)


def boot_ratio(a, b, n=2000, seed=0):
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(a), (n, len(a)))
    v = a[idx].sum(1) / b[idx].sum(1)
    return np.quantile(v, [.025, .975])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "oc_other.npz")
    d = np.load(path)
    dial = np.load(os.path.join(RES, "oc_dial.npz"))
    n_dev, n_te = len(d["dev_rows"]), len(d["te_rows"])

    # sanity against oc_dial.npz
    print("== sanity ==")
    for b, k in (("delete", "set11_delete"), ("morph", "set11_morph")):
        print(f"DEV {b}: max |KL - oc_dial| = {np.abs(d[f'dev_kl_{b}'] - dial[k][:n_dev]).max():.2e}")
        print(f"test {b}: max |KL - oc_dial| = {np.abs(d[f'te_kl_{b}'] - dial['te_' + k][:n_te]).max():.2e}")
    assert np.array_equal(d["te_rows"], dial["te_rows"][:n_te])

    # job units from DEV
    T = -d["dev_d_delete"].astype(np.float64)
    T -= T.mean(1, keepdims=True)
    e = (T ** 2).sum(0)
    rank = np.argsort(-e)
    cum = np.cumsum(e[rank]) / e.sum()
    sets = {q: rank[:int(np.searchsorted(cum, q)) + 1] for q in (.5, .8, .9)}
    print("\n== job set (DEV) ==")
    for q, J in sets.items():
        print(f"{int(q*100)}% of the children's effect: {len(J)} of {T.shape[1]} output units")
    flat = len(sets[.8]) / T.shape[1]

    # test photos
    lb = d["te_base"].astype(np.float64)
    top1 = lb.argmax(1)
    top5 = np.argsort(-lb, 1)[:, :5]
    print("\n== test photos: KL split, flips, other-capability photos ==")
    rows = {}
    for q, J in sets.items():
        inJ = np.zeros(lb.shape[1], bool); inJ[J] = True
        other_ph = ~inJ[top5].any(1)
        for b in BUILDS:
            ly = lb + d[f"te_d_{b}"].astype(np.float64)
            tot, mass, kin, kout = kl_split(lb, ly, J)
            new1 = ly.argmax(1)
            flip = new1 != top1
            oflip = flip & ~inJ[top1] & ~inJ[new1]
            dc = ly - lb; dc -= dc.mean(1, keepdims=True)
            e_out = (dc[:, ~inJ] ** 2).sum() / (dc ** 2).sum()
            rows[(q, b)] = dict(tot=tot, kout=kout, other_ph=other_ph, flip=flip, oflip=oflip, e_out=e_out)
            if q == .8 or b == "morph":
                print(f"J{int(q*100)} {b:9s}: KL {tot.mean():.5f} (check {np.abs(tot - d[f'te_kl_{b}']).max():.1e}) | "
                      f"mass {mass.mean():.5f} inJ {kin.mean():.5f} other {kout.mean():.5f} "
                      f"(other share {kout.sum()/tot.sum():.3f}) | logit energy off-job {e_out:.3f} | "
                      f"flips {flip.mean():.3f}, other flips {oflip.sum()} of {flip.sum()} | "
                      f"other-capability photos {other_ph.sum()}: KL {tot[other_ph].mean() if other_ph.any() else np.nan:.5f}, "
                      f"flips {flip[other_ph].mean() if other_ph.any() else np.nan:.3f}")

    # bars, J80, morph
    r = rows[(.8, "morph")]
    tot, kout, oph, flip, oflip = r["tot"], r["kout"], r["other_ph"], r["flip"], r["oflip"]
    print("\n== power ==")
    print(f"test photos {len(tot)}; se of mean KL left {tot.std()/np.sqrt(len(tot)):.5f} (mean {tot.mean():.5f})")
    print(f"other-capability photos (J80): {oph.sum()}" + ("  -> O2 NO POWER (< 100)" if oph.sum() < 100 else ""))
    print(f"job set J80 = {flat:.2f} of all output units" + ("  -> the job is NOT local in the output" if flat > .5 else ""))

    o1 = kout.sum() / tot.sum()
    ci1 = boot_ratio(kout, tot)
    o2k = tot[oph].mean() / tot.mean() if oph.any() else np.nan
    o2f = flip[oph].mean() if oph.any() else np.nan
    o3 = oflip.sum() / max(flip.sum(), 1)
    print("\n== bars (J80, morph, test) ==")
    print(f"O1 other KL share {o1:.3f} [{ci1[0]:.3f}, {ci1[1]:.3f}]  bar <= .10  -> {'pass' if o1 <= .10 else 'FAIL'}")
    if oph.sum() >= 100:
        ok2 = o2k <= .25 and o2f <= .01
        print(f"O2 other-capability photos: KL ratio {o2k:.3f} (bar <= .25), flips {o2f:.4f} (bar <= .01) -> "
              f"{'pass' if ok2 else 'FAIL'}")
    else:
        ok2 = None
        print(f"O2 no power ({oph.sum()} photos): KL ratio {o2k:.3f}, flips {o2f:.4f} (description only)")
    print(f"O3 other flips {oflip.sum()} of {flip.sum()} = {o3:.3f}  bar <= .10  -> {'pass' if o3 <= .10 else 'FAIL'}")
    if o1 > .10 or ok2 is False:
        base_other = kout.sum()
        for b in ("delmean", "morphmean"):
            ko = rows[(.8, b)]["kout"].sum()
            print(f"D1 {b}: other KL {ko/len(tot):.5f} vs morph {base_other/len(tot):.5f} -> cut {1 - ko/base_other:.2f} "
                  f"(bar >= .50) -> {'pass' if 1 - ko/base_other >= .5 else 'FAIL'}")

    print("\n== by parent peak (test, J80, morph; description) ==")
    pk = d["te_peak"]
    for a, b in zip([0, .5, 1, 1.5, 2, 2.92], [.5, 1, 1.5, 2, 2.92, 99]):
        s = (pk >= a) & (pk < b)
        if s.sum():
            print(f"{a}-{b}: n {s.sum():4d} KL left {tot[s].mean():.5f}, other {kout[s].mean():.5f} "
                  f"({kout[s].sum()/max(tot[s].sum(),1e-12):.2f}), flips {flip[s].mean():.3f}, other flips {oflip[s].sum()}")


if __name__ == "__main__":
    main()
