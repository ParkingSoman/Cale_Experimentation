# scale: neurons=A:block0-all; net=A, C-ladder, W-ladder, L-ladder, WC-ladder, R4C-ladder; photos=saved, broden-held; where=mac
# rung: Q=plan "Read what training adds on top of the calculation", steps 2+3: read the changes as concepts, widening test; N=weights + vecread dictionary; EST=3 min Mac; PERMIT=Shonu approved plan 2026-11-25
"""Steps 2 and 3 (Mac). Uses vecread (research/clean/vecread) on A's dictionary.

Step 2. Per cut, the change of the kept writes (summed over 3 x 3, bn2-scaled: the vector vecread reads as a write)
is a matrix (kept neurons x 512). Read: its top 3 singular directions (with variance share), the net sum, and the bn2
shift change. Changes: C-A, WC-C, W-A. Also the lost write (sum of D's A writes).
Floors: (a) K5b real-write spread (vr_real_floor.json), (b) 200 random Gaussian mixes of A's 512 summed writes.

Step 3. Widening test, bars written in the plan before the run:
  keeps job   a >= .8 (a = own-write coefficient in new write = a own + b partner + rest, 4608-dim effective writes)
  takes job   cos(write change, partner's A write) above the 95th percentile over all other deleted neurons
  read moves  cos(read change, partner's A read) above the same shuffled 95th percentile (L, R4C only)
  rough       the new write's top-8 concepts share >= 1 entry with the own top-8 AND >= 1 with the partner's top-8
Objects: n430 (n501), n76 (n374), n213 (n501) at 384; the top take-over unit of L at 11, 64, 192 (jobtest_L).
Output: results/oc_vs/read.md, read.json."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "vecread"))
import vecread as vr  # noqa: E402

LAD = os.path.join(HERE, "..", "results", "oc_ladder")
R18 = os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18")
OUT = os.path.join(HERE, "..", "results", "oc_vs")
EPS = 1e-5
RNG = np.random.default_rng(25)


def scale(bn):
    return bn[0] / np.sqrt(bn[3] + EPS)


def eff(w):
    """Effective writes (512 in x 4608), summed writes (512 x 512, row = neuron), reads (512 x 2304), bn2 shift."""
    s2 = scale(w["bn2_0"])
    V = (w["W2_0"] * s2[:, None, None, None]).astype(np.float64).transpose(1, 0, 2, 3)
    R = (w["W1_0"] * scale(w["bn1_0"])[:, None, None, None]).astype(np.float64).reshape(512, -1)
    t2 = w["bn2_0"][1] - w["bn2_0"][2] * s2
    return V.reshape(512, -1), V.sum((2, 3)), R, t2.astype(np.float64)


def brief(card, k=3):
    return ", ".join(f"{n} {s:+.2f}" for n, _, s in card["up"][:k]) + f" | down: " + \
        ", ".join(f"{n} {s:+.2f}" for n, _, s in card["down"][:2]) + f" | left {card['leftover']:.2f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    dic = vr.Dictionary("s4b0")
    dk = vr.Dictionary("s3k")
    floor = json.load(open(os.path.join(HERE, "..", "..", "vecread", "results", "vr_real_floor.json")))
    A = np.load(os.path.join(R18, "w_A.npz"))
    VA, SA, RA, tA = eff(A)
    # random-mix floor: Gaussian mixes of A's summed writes
    rnd = [vr.read_vector(dic, RNG.standard_normal(512) @ SA, "write") for _ in range(200)]
    rtop = np.array([c["up"][0][2] for c in rnd])
    rleft = np.array([c["leftover"] for c in rnd])
    md = ["# Steps 2 and 3: vector reads of what training adds (oc_vs_read.py)", "",
          f"Floors. Random mixes of A's writes (200): top cosine 5th/50th/95th "
          f"{np.percentile(rtop, 5):.2f} / {np.median(rtop):.2f} / {np.percentile(rtop, 95):.2f}; leftover "
          f"{np.percentile(rleft, 5):.2f} / {np.median(rleft):.2f} / {np.percentile(rleft, 95):.2f}. "
          f"A's real writes (K5b): top cosine 5th/50th/95th "
          f"{' / '.join(f'{x:.2f}' for x in floor['write_s4b0']['top_score_real'].values())}; "
          f"leftover {' / '.join(f'{x:.2f}' for x in floor['write_s4b0']['leftover_real'].values())}.", ""]
    res = dict(floor=dict(rand_top=rtop.tolist(), rand_left=rleft.tolist()), cuts={})
    for n in (11, 64, 192, 384):
        d = os.path.join(LAD, f"n{n}")
        jt = np.load(os.path.join(d, "jobtest_L.npz"))
        D, K = jt["D"], jt["K"]
        W = {b: eff(np.load(os.path.join(d, f"w_{b}.npz"))) for b in ("C", "W", "WC", "L", "R4C")}
        W["A"] = (VA, SA, RA, tA)
        r = {}
        md += [f"## Cut {n}", "", "| Vector | Share | Reading (up 3, down 2, leftover) |", "|---|---|---|"]
        lost = SA[D].sum(0)
        c = vr.read_vector(dic, lost, "write")
        r["lost"] = c
        md.append(f"| lost: sum of D's writes | | {brief(c)} |")
        for x, ref in (("C", "A"), ("WC", "C"), ("W", "A")):
            M = W[x][1][K] - W[ref][1][K]                                  # kept x 512 change of summed writes
            u, s, vt = np.linalg.svd(M, full_matrices=False)
            share = s ** 2 / (s ** 2).sum()
            for i in range(3):
                v = vt[i] * np.sign(u[:, i].sum() or 1)
                c = vr.read_vector(dic, v, "write")
                r[f"{x}-{ref}_sv{i}"] = dict(card=c, share=float(share[i]))
                md.append(f"| {x}-{ref} direction {i + 1} | {share[i]:.2f} | {brief(c)} |")
            net = M.sum(0)
            c = vr.read_vector(dic, net, "write")
            r[f"{x}-{ref}_net"] = c
            md.append(f"| {x}-{ref} net sum | | {brief(c)} |")
            dt = W[x][3] - W[ref][3]
            if np.linalg.norm(dt) > 1e-6:
                c = vr.read_vector(dic, dt, "write")
                r[f"{x}-{ref}_shift"] = dict(card=c, norm=float(np.linalg.norm(dt)))
                md.append(f"| {x}-{ref} bn2 shift change (norm {np.linalg.norm(dt):.3f}) | | {brief(c)} |")
        # direction overlap: WC-C top directions against C-A top directions
        Mc = W["C"][1][K] - SA[K]
        Mt = W["WC"][1][K] - W["C"][1][K]
        vc = np.linalg.svd(Mc, full_matrices=False)[2][:10]
        vt_ = np.linalg.svd(Mt, full_matrices=False)[2][:10]
        ov = float((np.linalg.svd(vc @ vt_.T, compute_uv=False) ** 2).mean())
        r["overlap_top10_WC-C_vs_C-A"] = ov
        cn = float(Mt.sum(0) @ Mc.sum(0) / (np.linalg.norm(Mt.sum(0)) * np.linalg.norm(Mc.sum(0))))
        r["cos_net_WC-C_vs_C-A"] = cn
        md += [f"Cosine of the net sums, WC-C against C-A: **{cn:+.3f}** (negative = training pulls back C's rebuild)", ""]
        md += ["", f"Overlap of the top-10 direction spaces, WC-C against C-A (mean squared cosine of principal angles; "
                   f"random 10-dim pair in 512: ~{10 / 512:.3f}): **{ov:.3f}**", ""]

        # ---- step 3: widening ----
        if n == 384:
            pairs = [(430, 501), (76, 374), (213, 501)]
        else:
            i = int(np.argmax(np.where(jt["alive_A"], jt["take"], -9)))
            pairs = [(int(K[i]), int(D[jt["best_d"][i]]))]
        md += ["| Unit -> partner | Build | a (own) | write change: partner cos, pct among D | read-pattern change: partner cos, pct | rough: shared own / partner | verdict |",
               "|---|---|---|---|---|---|---|"]
        wide = []
        for j, p in pairs:
            ownc = vr.read_vector(dic, SA[j], "write")
            parc = vr.read_vector(dic, SA[p], "write")
            for b in ("C", "WC", "L", "R4C"):
                Vn, Sn, Rn, _ = W[b]
                X = np.stack([VA[j], VA[p]], 1)
                a, bb = np.linalg.lstsq(X, Vn[j], rcond=None)[0]
                dw = Vn[j] - VA[j]
                cosd = np.array([dw @ VA[dd] / (np.linalg.norm(dw) * np.linalg.norm(VA[dd]) + 1e-30) for dd in D])
                ip = int(np.nonzero(D == p)[0][0])
                pct_w = float((cosd < cosd[ip]).mean())
                dr = Rn[j] - RA[j]
                if np.linalg.norm(dr) > 1e-9:
                    # reads compared as patterns (cov @ r, Haufe; vecread rule), whitened metric as vecread reads them
                    gp = dk.metric(dk.pattern(dr), True)[0]
                    GD = np.stack([dk.metric(dk.pattern(RA[dd]), True)[0] for dd in D])
                    cr = GD @ gp / (np.linalg.norm(GD, axis=1) * np.linalg.norm(gp) + 1e-30)
                    pct_r, cr_p = float((cr < cr[ip]).mean()), float(cr[ip])
                else:
                    pct_r, cr_p = None, None
                newc = vr.read_vector(dic, Sn[j], "write")
                names = lambda c: {x[0] for x in c["up"]}
                sh_o, sh_p = len(names(newc) & names(ownc)), len(names(newc) & names(parc))
                bars = dict(keeps=a >= .8, takes=pct_w >= .95, read=(pct_r is not None and pct_r >= .95),
                            rough=sh_o >= 1 and sh_p >= 1)
                verdict = "coarse (all 4)" if all(bars.values()) else "fails: " + ", ".join(k for k, v in bars.items() if not v)
                wide.append(dict(unit=j, partner=p, build=b, a=float(a), b=float(bb), cos_w=float(cosd[ip]), pct_w=pct_w,
                                 cos_r=cr_p, pct_r=pct_r, shared_own=sh_o, shared_partner=sh_p,
                                 bars={k: bool(v) for k, v in bars.items()}, new_top=newc["up"][:5]))
                rr = "no read change" if pct_r is None else f"{cr_p:+.3f}, {pct_r:.2f}"
                md.append(f"| n{j} -> n{p} | {b} | {a:.3f} | {cosd[ip]:+.3f}, {pct_w:.2f} | {rr} | {sh_o} / {sh_p} | {verdict} |")
            md += [f"", f"n{j} own write: {brief(ownc)}", f"", f"n{p} partner write: {brief(parc)}", ""]
        r["widening"] = wide
        res["cuts"][str(n)] = r
        md.append("")
        print(f"cut {n} done", flush=True)
    json.dump(res, open(os.path.join(OUT, "read.json"), "w"), indent=1, default=str)
    open(os.path.join(OUT, "read.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
