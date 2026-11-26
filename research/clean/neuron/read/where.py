"""N10 step 1: where the gap is, by WordNet family (true label), and where B's and T's extra probability goes."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "results", "read")
sys.path.insert(0, HERE)
from families import syn, ancestors
sys.path.insert(0, os.path.join(HERE, "..", "gap")); import common as C
FAM = np.array(json.load(open(os.path.join(OUT, "families.json")))["families"])
NAMES = sorted(set(FAM))
LAB = C.net.Photos("read").lab
rng = np.random.default_rng(0); NB = 2000

def near_matrix():
    f = os.path.join(OUT, "near.npy")
    if os.path.exists(f): return np.load(f)
    up2, anc = [], []
    for i in range(1000):
        s = syn(i); a2 = set()
        for p in s.hypernym_paths(): a2 |= set(p[-3:])          # the class and up to 2 steps above
        up2.append(a2); anc.append(ancestors(s))
    M = np.array([[bool(up2[i] & anc[j]) for j in range(1000)] for i in range(1000)])
    np.save(f, M); return M

def sm(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def kl(p, q): return (p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum(1)

def boot(fn, n, idx_groups=None):
    vals = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    return np.percentile(vals, [2.5, 97.5]).tolist()

def run(s):
    d = np.load(os.path.join(OUT, f"logits_{s}.npz"))
    p, qB, qT = sm(d["base"].astype(np.float64)), sm(d["B"].astype(np.float64)), sm(d["T"].astype(np.float64))
    y = LAB[d["ix"]]; fam = FAM[y]; n = len(y)
    klB, klT = kl(p, qB), kl(p, qT); gap = klB - klT
    infam = FAM[None, :] == fam[:, None]                       # (n, 1000) class in the photo's family
    NEAR = near_matrix()[y]                                    # (n, 1000)
    top = p.argmax(1); offtop = p.copy(); offtop[np.arange(n), top] = 0
    out = dict(n=n, klB=klB.mean(), klT=klT.mean(), gap=gap.mean(), fam={})
    # family table
    for f in NAMES:
        m = fam == f
        if not m.any(): continue
        share = gap[m].sum() / gap.sum()
        sh_ci = boot(lambda r: gap[r][m[r]].sum() / gap[r].sum(), n)
        g = gap[m]; rate_ci = np.percentile([g[rng.integers(0, len(g), len(g))].mean() for _ in range(NB)], [2.5, 97.5]).tolist()
        out["fam"][f] = dict(photos=int(m.sum()), photo_share=m.mean(), share=share, share_ci=sh_ci, rate=g.mean(), rate_ci=rate_ci,
                             klB=klB[m].mean(), klT=klT[m].mean())
    # where the extra probability goes
    def dest(q, mask):
        gm = np.clip(q - p, 0, None)
        G = gm.sum(1); r = lambda M, rows: (gm[rows] * M[rows]).sum() / G[rows].sum()
        res = {}
        for name, M in (("near", NEAR), ("family", infam), ("family_not_near", infam & ~NEAR)):
            rows = np.where(mask)[0]
            res[name] = r(M, rows); res[name + "_ci"] = boot(lambda b: r(M, rows[b]), len(rows))
        res["G"] = G[mask].mean()
        return res
    def ctrl(mask):
        rows = np.where(mask)[0]
        r = lambda M, rr: (offtop[rr] * M[rr]).sum() / offtop[rr].sum()
        return dict(near=r(NEAR, rows), family=r(infam, rows), size_family=infam[rows].mean(), size_near=NEAR[rows].mean())
    out["dest"] = {}
    for f in ["ALL"] + NAMES:
        mask = np.ones(n, bool) if f == "ALL" else fam == f
        if mask.sum() < 20: continue
        out["dest"][f] = dict(B=dest(qB, mask), T=dest(qT, mask), base_ctrl=ctrl(mask))
    # where B's out-of-family extra mass goes, by destination family, for each source family
    gB = np.clip(qB - p, 0, None)
    out["out_dest"] = {}
    for f in NAMES:
        m = fam == f
        if m.sum() < 20: continue
        tot = {g: float(gB[m][:, FAM == g].sum()) for g in NAMES if g != f}
        Z = sum(tot.values()); out["out_dest"][f] = sorted(((g, v / Z) for g, v in tot.items()), key=lambda t: -t[1])[:4]
    return out

def conv(o):
    if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
if __name__ == "__main__":
    sets = sys.argv[1:] or ["held"]
    for s in sets:
        o = run(s); json.dump(conv(o), open(os.path.join(OUT, f"where_{s}.json"), "w"), indent=1)
        print(f"== {s}: n {o['n']}  KL B {o['klB']:.4f} T {o['klT']:.4f} gap {o['gap']:.4f}")
        print(f"{'family':22s} {'photos':>6s} {'p.sh':>5s} {'share':>6s} {'share 95%':>15s} {'rate':>6s} {'rate 95%':>15s} {'KL B':>6s} {'KL T':>6s}")
        for f, r in sorted(o["fam"].items(), key=lambda t: -t[1]["share"]):
            print(f"{f:22s} {r['photos']:6d} {r['photo_share']:5.3f} {r['share']:6.3f} [{r['share_ci'][0]:.3f},{r['share_ci'][1]:.3f}] {r['rate']:6.3f} [{r['rate_ci'][0]:.3f},{r['rate_ci'][1]:.3f}] {r['klB']:6.3f} {r['klT']:6.3f}")
        print(f"{'dest':22s} {'B near':>7s} {'B fam':>7s} {'B fam 95%':>15s} {'T near':>7s} {'T fam':>7s} {'base nr':>7s} {'base fam':>8s} {'size fam':>8s} {'G B':>5s} {'G T':>5s}")
        for f, r in o["dest"].items():
            B, T, c = r["B"], r["T"], r["base_ctrl"]
            print(f"{f:22s} {B['near']:7.3f} {B['family']:7.3f} [{B['family_ci'][0]:.3f},{B['family_ci'][1]:.3f}] {T['near']:7.3f} {T['family']:7.3f} {c['near']:7.3f} {c['family']:8.3f} {c['size_family']:8.3f} {B['G']:5.3f} {T['G']:5.3f}")
        for f, v in o["out_dest"].items(): print(f"  B's out-of-family extra from {f}: " + ", ".join(f"{g} {x:.2f}" for g, x in v))
