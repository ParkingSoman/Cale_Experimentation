# scale: neurons=A:block0-all; net=A, Z-ladder, C-ladder, WC-ladder, L-ladder, R4C-ladder; photos=test; where=mac
"""Mac reader for oc_vs_jobmap.py (plan "Where did each deleted neuron's job go?", approved 2026-11-25).

Per cut and build, per deleted unit d (only d with a real repair: R[d] >= .1 KL(Z) on d's photos):
  g[j, d]  = mean over d's 50 photos of KL(m, j swapped back) - KL(m)
  top-1 share = max positive g / sum positive g;  N80 = fewest units holding 80% of positive g;  N_eff
  specific = the top unit's g on d's photos >= 3x its mean g on the 500 random photos
  verdict: One (share >= .5 and specific), Few (N80 <= 5), Many (N80 > 5)
Usage: python3 oc_vs_jobmap_read.py [results folder]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "oc_vs_jobmap")
BUILDS = ("C", "WC", "L", "R4C")


def per_d(M, top_u, rand_u):
    g = np.stack([M[:, top_u[i]].astype(np.float64).mean(1) for i in range(len(top_u))], 1)   # K x D
    gr = M[:, rand_u].astype(np.float64).mean(1)                                               # K
    return g, gr


def stats(g, gr, R, ok):
    out = []
    for i in range(g.shape[1]):
        col = g[:, i]
        gp = np.clip(col, 0, None)
        P = gp.sum()
        srt = np.sort(gp)[::-1]
        top = int(np.argmax(col))
        share = float(srt[0] / P) if P > 0 else 0.0
        n80 = int(np.searchsorted(np.cumsum(srt), .8 * P) + 1) if P > 0 else 0
        neff = float(P ** 2 / (gp ** 2).sum()) if P > 0 else 0.0
        spec = bool(col[top] >= 3 * max(gr[top], 0) and col[top] > 0)
        if share >= .5 and spec:
            v = "One"
        elif n80 <= 5:
            v = "Few"
        else:
            v = "Many"
        out.append(dict(top=top, share1=share, n80=n80, neff=neff, specific=spec, verdict=v,
                        sum_over_R=float(col.sum() / R[i]) if R[i] > 0 else None, counted=bool(ok[i]),
                        top_g=float(col[top]), top_g_rand=float(gr[top])))
    return out


def main():
    rep = {}
    lines = ["# Where did each deleted neuron's job go?", ""]
    for f in sorted(os.listdir(RES), key=lambda s: int(s[1:-4]) if s.startswith("n") and s.endswith(".npz") else 0):
        if not (f.startswith("n") and f.endswith(".npz")):
            continue
        z = np.load(os.path.join(RES, f))
        n = int(f[1:-4])
        D, K, top_u, rand_u, klZ = z["D"], z["K"], z["top_u"], z["rand_u"], z["klZ_u"]
        Zd = klZ[top_u].mean(1)
        rep[n] = {}
        lines += [f"## cut {n} ({len(D)} deleted, {len(K)} kept)", "",
                  "| build | counted d | One | Few | Many | verdict | median top-1 share | median N80 | median sum g / R |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for b in BUILDS:
            if f"{b}_full" not in z.files:
                continue
            base = z[f"{b}_base_u"]
            R = Zd - base[top_u].mean(1)
            ok = R >= .1 * Zd
            g, gr = per_d(z[f"{b}_full"], top_u, rand_u)
            st = stats(g, gr, R, ok)
            for i, s in enumerate(st):
                s["d"] = int(D[i])
                s["top_unit"] = int(K[s["top"]])
                for v in ("read", "write"):
                    if f"{b}_{v}" in z.files:
                        gv, _ = per_d(z[f"{b}_{v}"][[s["top"]]], top_u[i:i + 1], rand_u)
                        s[f"top_g_{v}"] = float(gv[0, 0])
            c = [s for s in st if s["counted"]]
            cnt = {v: sum(s["verdict"] == v for s in c) for v in ("One", "Few", "Many")}
            nc = max(len(c), 1)
            verdict = "merge" if cnt["One"] >= nc / 2 else ("spread" if cnt["Many"] >= nc / 2 else "mixed")
            med = lambda k: float(np.median([s[k] for s in c if s[k] is not None])) if c else float("nan")
            lines.append(f"| {b} | {len(c)}/{len(D)} | {cnt['One']} | {cnt['Few']} | {cnt['Many']} | {verdict} | "
                         f"{med('share1'):.2f} | {med('n80'):.0f} | {med('sum_over_R'):.2f} |")
            rep[n][b] = dict(per_d=st, counts=cnt, verdict=verdict, n_counted=len(c))
        lines.append("")
        # named rows
        named = [int(d) for d in D] if n == 11 else ([476] if 476 in D else [])
        for b in ("WC", "L"):
            if b not in rep[n]:
                continue
            rows = [s for s in rep[n][b]["per_d"] if s["d"] in named]
            if not rows:
                continue
            lines += [f"Named, {b}: deleted d -> top kept unit (share, N80, verdict, g on d / g random, "
                      "g read-only / write-only)", ""]
            for s in rows:
                rw = (f", read {s.get('top_g_read', float('nan')):.2e} / write {s.get('top_g_write', float('nan')):.2e}"
                      if "top_g_read" in s else "")
                lines.append(f"- n{s['d']} -> n{s['top_unit']}: share {s['share1']:.2f}, N80 {s['n80']}, "
                             f"{s['verdict']}{'' if s['counted'] else ' (not counted)'}, "
                             f"g {s['top_g']:.2e} / {s['top_g_rand']:.2e}{rw}")
            lines.append("")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(RES, "jobmap_read.md"), "w").write(txt + "\n")
    json.dump(rep, open(os.path.join(RES, "jobmap_read.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
