# scale: neurons=A:block0-all; net=A, Z-ladder, C-ladder, W-ladder, L-ladder, R4-ladder, R-ladder, WC-ladder, R4C-ladder; photos=saved, test; where=mac
# rung: Q=plan "Read what training adds on top of the calculation", step 4: does a repair hurt unrelated jobs; N=saved per-photo KL; EST=10 s Mac; PERMIT=Shonu approved plan 2026-11-25
"""Step 4 (Mac, saved per-photo KL on the 17,000 test photos). Per cut and build:
  recovery on all photos, on the quiet half (bottom 50% by A's summed firing of D) and the busy tenth (top 10%);
  share of photos worse than Z (delete only), all and quiet; classes (of 1000) whose mean KL is above Z's.
A repair that is worse than deleting on the quiet photos hurts jobs the deletion did not touch.
Inputs: results/oc_ladder/n<cut>/kl_test_<build>.npy, results/oc_vs/n<cut>.npz (dfire, labels; from oc_vs_why.py).
95% CI on quiet recovery: paired bootstrap over photos (2,000)."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAD = os.path.join(HERE, "..", "results", "oc_ladder")
OUT = os.path.join(HERE, "..", "results", "oc_vs")
RNG = np.random.default_rng(4)


def rec(k, z, idx):
    return 1 - k[idx].mean() / z[idx].mean()


def main():
    res, md = {}, ["# Step 4: harm to unrelated jobs (oc_vs_harm.py)", ""]
    for n in (11, 64, 192, 384):
        aux = np.load(os.path.join(OUT, f"n{n}.npz"))
        fire, lab = aux["dfire"], aux["labels"]
        z = np.load(os.path.join(LAD, f"n{n}", "kl_test_Z.npy"))
        assert abs(z.mean() - aux["Z__kl"].mean()) < 1e-4 * z.mean(), "photo order differs from the ladder"
        quiet = fire <= np.quantile(fire, .5)
        busy = fire >= np.quantile(fire, .9)
        boot = [RNG.integers(0, quiet.sum(), quiet.sum()) for _ in range(2000)]
        qi = np.nonzero(quiet)[0]
        cls = np.unique(lab)
        zc = np.array([z[lab == c].mean() for c in cls])
        md += [f"## Cut {n} (KL of Z: all {z.mean():.4f}, quiet {z[quiet].mean():.4f}, busy {z[busy].mean():.4f})", "",
               "| Build | Rec all | Rec quiet [95% CI] | Rec busy | Worse than Z: all / quiet | Classes worse than Z |",
               "|---|---|---|---|---|---|"]
        r = {}
        for b in ("C", "W", "WC", "L", "R4", "R4C", "R"):
            f = os.path.join(LAD, f"n{n}", f"kl_test_{b}.npy")
            if not os.path.exists(f):
                continue
            k = np.load(f)
            ci = np.percentile([1 - k[qi[i]].mean() / z[qi[i]].mean() for i in boot], [2.5, 97.5])
            kc = np.array([k[lab == c].mean() for c in cls])
            worse_cls = int((kc > zc).sum())
            row = dict(rec_all=float(rec(k, z, slice(None))), rec_quiet=float(rec(k, z, quiet)),
                       rec_quiet_ci=ci.tolist(), rec_busy=float(rec(k, z, busy)),
                       worse_all=float((k > z).mean()), worse_quiet=float((k[quiet] > z[quiet]).mean()),
                       classes_worse=worse_cls,
                       top_worse_classes=[(int(cls[i]), float(kc[i] - zc[i])) for i in np.argsort(zc - kc)[:5]
                                          if kc[i] > zc[i]])
            r[b] = row
            md.append(f"| {b} | {row['rec_all']:.3f} | {row['rec_quiet']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                      f"{row['rec_busy']:.3f} | {row['worse_all']:.2f} / {row['worse_quiet']:.2f} | {worse_cls} |")
        res[str(n)] = r
        md.append("")
    json.dump(res, open(os.path.join(OUT, "harm.json"), "w"), indent=1)
    open(os.path.join(OUT, "harm.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
