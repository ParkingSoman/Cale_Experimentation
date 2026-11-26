# scale: neurons=s3:69, A:children; net=A; photos=saved; where=mac
"""Step 2 of LOG "Does coarsening hurt other capabilities?": the job as output directions. numpy only.

S_q = the fewest right singular vectors of the DEV matrix of the children's effect T (base - delete, centred over
units, NOT centred over photos) that hold q of its energy. On test photos: leftover E = morph - base (centred);
E_out = E minus its projection on S80; off-job KL = KL(base || base + E_out); off-job flip = top-1 change by E_out.
Usage: python3 oc_other_dir.py
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")


def cen(a):
    a = a.astype(np.float64)
    return a - a.mean(1, keepdims=True)


def logsm(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def kl(lb, ly):
    a, b = logsm(lb), logsm(ly)
    return (np.exp(a) * (a - b)).sum(1)


def main():
    d = np.load(os.path.join(RES, "oc_other.npz"))
    T = -cen(d["dev_d_delete"])
    _, s, vt = np.linalg.svd(T, full_matrices=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()
    ks = {q: int(np.searchsorted(cum, q)) + 1 for q in (.5, .8, .9)}
    print("== job directions (DEV) ==")
    for q, k in ks.items():
        print(f"S{int(q*100)}: {k} directions (random subspace of this size holds {k/1000:.3f})")

    lb = d["te_base"].astype(np.float64)
    top1 = lb.argmax(1)
    Tt = -cen(d["te_d_delete"])
    klm = d["te_kl_morph"].astype(np.float64)
    flip_m = (lb + d["te_d_morph"].astype(np.float64)).argmax(1) != top1
    print(f"\n== test (2,000): KL left by morph {klm.mean():.5f} (se {klm.std()/np.sqrt(len(klm)):.5f}), "
          f"flips {flip_m.sum()} ==")
    out = {}
    for q, k in ks.items():
        V = vt[:k]

        def off(x):
            return x - (x @ V.T) @ V
        t_off = (off(Tt) ** 2).sum() / (Tt ** 2).sum()
        rows = {}
        for b in ("delete", "morph", "morphmean"):
            E = cen(d[f"te_d_{b}"])
            Eo = off(E)
            ko = kl(lb, lb + Eo)
            fo = (lb + Eo).argmax(1) != top1
            fb = (lb + d[f"te_d_{b}"].astype(np.float64)).argmax(1) != top1
            rows[b] = dict(e_off=(Eo ** 2).sum() / (E ** 2).sum(), ko=ko, fo=fo, fb=fb,
                           kb=d[f"te_kl_{b}"].astype(np.float64))
            print(f"S{int(q*100)} {b:9s}: off-S energy {rows[b]['e_off']:.3f} (T's own {t_off:.3f}) | "
                  f"off-job KL {ko.mean():.5f} = {ko.mean()/rows[b]['kb'].mean():.3f} of KL left | "
                  f"off-job flips {fo.sum()} of {fb.sum()}")
        out[q] = (t_off, rows)

    t_off, r = out[.8]
    m = r["morph"]
    boot = np.random.default_rng(0).integers(0, len(klm), (2000, len(klm)))
    ci = np.quantile(m["ko"][boot].mean(1) / m["kb"][boot].mean(1), [.025, .975])
    print("\n== bars (S80, morph, test) ==")
    print(f"Q0 S80 directions {ks[.8]} (bar <= 50) -> {'pass' if ks[.8] <= 50 else 'FAIL'}")
    q1 = m["ko"].mean() / m["kb"].mean()
    print(f"Q1 off-job KL share {q1:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (bar <= .10) -> {'pass' if q1 <= .10 else 'FAIL'}")
    q2 = m["fo"].sum() / max(m["fb"].sum(), 1)
    print(f"Q2 off-job flips {m['fo'].sum()} of {m['fb'].sum()} = {q2:.3f} (bar <= .10) -> "
          f"{'pass' if q2 <= .10 else 'FAIL'}")
    print(f"Q3 off-S80 energy of E {m['e_off']:.3f} vs T's own {t_off:.3f} -> {'pass' if m['e_off'] <= t_off else 'FAIL'}")

    print("\n== by parent peak (S80, morph; description) ==")
    pk = d["te_peak"]
    for a, b in zip([0, .5, 1, 1.5, 2, 2.92], [.5, 1, 1.5, 2, 2.92, 99]):
        s_ = (pk >= a) & (pk < b)
        print(f"{a}-{b}: n {s_.sum():4d} KL left {m['kb'][s_].mean():.5f}, off-job KL {m['ko'][s_].mean():.5f} "
              f"({m['ko'][s_].mean()/m['kb'][s_].mean():.2f}), off-job flips {m['fo'][s_].sum()}")


if __name__ == "__main__":
    main()
