"""N10 step 1 check: in-family / near shares with the true class AND Base's top class removed from both sides;
and Base's own out-of-family spread by destination family (control for the 'sink' reading)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from where import FAM, NAMES, LAB, near_matrix, sm, OUT
rng = np.random.default_rng(1)
def run(s):
    d = np.load(os.path.join(OUT, f"logits_{s}.npz"))
    p, qB, qT = (sm(d[k].astype(np.float64)) for k in ("base", "B", "T"))
    y = LAB[d["ix"]]; fam = FAM[y]; n = len(y); r = np.arange(n)
    keep = np.ones((n, 1000), bool); keep[r, y] = False; keep[r, p.argmax(1)] = False
    infam = FAM[None, :] == fam[:, None]; NEAR = near_matrix()[y]
    off = p * keep
    out = {}
    def share(w, M, rows): return (w[rows] * M[rows]).sum() / w[rows].sum()
    for f in ("ALL", "birds"):
        rows = r if f == "ALL" else r[fam == f]
        o = {}
        for nm, q in (("B", qB), ("T", qT)):
            g = np.clip(q - p, 0, None) * keep
            o[nm] = {k: share(g, M, rows) for k, M in (("family", infam), ("near", NEAR))}
            o[nm]["family_ci"] = np.percentile([share(g, infam, rows[rng.integers(0, len(rows), len(rows))]) for _ in range(1000)], [2.5, 97.5]).tolist()
            o[nm]["G"] = g[rows].sum(1).mean()
        o["base"] = {k: share(off, M, rows) for k, M in (("family", infam), ("near", NEAR))}
        out[f] = o
    # sink control: out-of-family destination shares, B's extra vs Base's own off-top spread, per source family
    gB = np.clip(qB - p, 0, None) * keep
    sink = {}
    for f in NAMES:
        m = fam == f
        if m.sum() < 20: continue
        oth = FAM != f
        b = {g: gB[m][:, FAM == g].sum() / gB[m][:, oth].sum() for g in ("dogs", "devices+instruments")}
        c = {g: off[m][:, FAM == g].sum() / off[m][:, oth].sum() for g in ("dogs", "devices+instruments")}
        sink[f] = dict(B=b, base=c)
    out["sink"] = sink
    return out
if __name__ == "__main__":
    for s in ("held", "test"):
        o = run(s); json.dump(o, open(os.path.join(OUT, f"where_check_{s}.json"), "w"), indent=1, default=float)
        for f in ("ALL", "birds"):
            B, T, b = o[f]["B"], o[f]["T"], o[f]["base"]
            print(f"{s} {f:6s} in-family B {B['family']:.3f} [{B['family_ci'][0]:.3f},{B['family_ci'][1]:.3f}] T {T['family']:.3f} base {b['family']:.3f} | near B {B['near']:.3f} T {T['near']:.3f} base {b['near']:.3f} | G B {B['G']:.3f} T {T['G']:.3f}")
        if s == "held":
            print("sink: share of out-of-family extra (B) vs Base's own out-of-family spread, to dogs / to devices")
            for f, v in o["sink"].items():
                print(f"  {f:22s} dogs B {v['B']['dogs']:.2f} base {v['base']['dogs']:.2f} | devices B {v['B']['devices+instruments']:.2f} base {v['base']['devices+instruments']:.2f}")
