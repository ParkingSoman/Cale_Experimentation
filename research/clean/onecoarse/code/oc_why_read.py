# scale: neurons=s3:69, A:children; net=A; photos=saved; where=mac
"""Read oc_why.npz (LOG "Why does the trained write leave the children's paths?"). Mac numpy.
Effective 7 x 7 kernel of a factored write (k: 11 x 1 x 3 x 3 stride 2, v: 512 x 11 x 3 x 3):
E[o, 2da+e, 2db+f] += sum_h v[o, h, da, db] k[h, 0, e, f]. W1: cos(W_LS, trained). W2: recoveries. W3: DEV weak range M.O/O.O."""
import numpy as np

d = np.load("results/oc_why.npz"); mo = np.load("results/oc_morph2.npz")


def eff(k, v):
    E = np.zeros((512, 7, 7))
    for da in range(3):
        for db in range(3):
            for e in range(3):
                for f in range(3):
                    E[:, 2 * da + e, 2 * db + f] += v[:, :, da, db] @ k[:, 0, e, f]
    return E


def cos(a, b):
    return float(a.ravel() @ b.ravel() / np.linalg.norm(a) / np.linalg.norm(b))


C, T0, T1, L = eff(mo["k_cal"], mo["v_cal"]), eff(mo["tr0_k"], mo["tr0_v"]), eff(mo["tr1_k"], mo["tr1_v"]), d["W_ls"][:, 0]
print(f"cos: LS-tr0 {cos(L, T0):.2f}, LS-tr1 {cos(L, T1):.2f}, calc-tr0 {cos(C, T0):.2f}, calc-LS {cos(C, L):.2f}, tr0-tr1 {cos(T0, T1):.2f}")
print(f"norms: calc {np.linalg.norm(C):.2f}, tr0 {np.linalg.norm(T0):.2f}, LS {np.linalg.norm(L):.2f}")
for name in ("test_strong", "test_random"):
    D = d[f"{name}_delete"]
    print(name, {b: round(1 - d[f'{name}_{b}'].mean() / D.mean(), 3) for b in ("calc", "tr0", "tr1", "ls")})
pk = d["dev_peak"]
for lo, hi in [(.5, 2), (2, 99), (0, 99)]:
    m = (pk >= lo) & (pk < hi)

    def s(k):
        return d[f"dev_{k}"][m].sum()
    row = {b: (round(s(f'{b}_MRo') / s('RoRo'), 2), round(s(f'{b}_MO') / s('OO'), 2),
               round(1 - (s('TT') - 2 * s(f'{b}_MT') + s(f'{b}_MM')) / s('TT'), 2)) for b in ("calc", "tr0", "tr1", "ls")}
    print(f"DEV peak {lo}-{hi} (n {m.sum()}): (on route, on condition part, logit recovery) {row}")
