"""CALC 3.6, Step 4 number tests (plan id-like-you-to-fizzy-cherny, 2026-11-25). numpy only, from out/poles.npz.
Rules fixed here BEFORE the full poles.npz existed. Crops are FIT crops (positions 0-3), 40 per pole.

T3 (coarse neuron does not split its branch). Per group: (a) the student's median pre-activation on each member's
   top crops; (b) each member's median pre-activation on the student's negative crops. A group SPLITS if any (a) < 0
   or any (b) > 0. Prediction: fewer than 3 of 10 groups split. Context null: the student on random-pool top crops.
T2 (siblings gate each other). Ordered pairs (i, j) of members in the same block and group. x = i's median
   pre-activation on j's top crops. Null = i's medians on the top crops of the 40 pool neurons of j's block.
   A pair GATES if x is below the null's 10th percentile. Prediction: at least half the pairs gate.
T4 (siblings share a parent). Top-10 feeders of a member = the 10 largest mean contributions on its top crops,
   counting attention heads and MLP neurons only (no embedding, no bias terms). Shared = mean over member pairs of
   |top10_i & top10_j|. Null: 1,000 triples; each member is replaced by a pool neuron of the same block, drawn from
   the 5 pool neurons nearest in FIT mean max firing. Pass for a group if shared > null 95th percentile.
   Prediction: at least 6 of 10 groups pass.
T6 (parent or sibling, description). Where members sit in blocks b-1 and b: rank of the b-1 member among the b
   member's MLP feeders.
Gate numbers (checklist step 3): for each named unit, the summed contribution of its top-10 up-voting feeders at its
   top crops and at its negative crops, and its 5 strongest down-voting feeders at its negative crops.
Output: out/tree_verbs.json
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def main(fn="poles.npz"):
    P = np.load(os.path.join(OUT, fn))
    role, grp, blk, neu = P["unit_role"], P["unit_group"], P["unit_block"], P["unit_neuron"]
    crops, cross = P["crops"], P["cross"]
    fit = P["pos"] < 4
    NU = len(role)
    top = {k: crops[(crops[:, 0] == k) & (crops[:, 1] == 0)] for k in range(NU)}
    neg = {k: crops[(crops[:, 0] == k) & (crops[:, 1] == 1)] for k in range(NU)}

    row_of = {tuple(c): i for i, c in enumerate(crops)}

    def cs_index(cs):
        return np.array([row_of[tuple(c)] for c in cs])

    def med(k, cs):  # unit k's median pre-activation on the crops cs
        return float(np.median(cross[cs_index(cs), k]))

    pool = {b: np.where((role == "random") & (blk == b))[0] for b in np.unique(blk[role == "random"])}
    groups = sorted(set(grp[grp >= 0]))
    R = {"T3": [], "T2": [], "T4": [], "T6": [], "gate": {}}

    # ---- T3 ----
    for g in groups:
        s = np.where((grp == g) & (role == "student"))[0][0]
        mem = np.where((grp == g) & (role != "student"))[0]
        a = {str(role[m]): med(s, top[m]) for m in mem}
        b = {str(role[m]): med(m, neg[s]) for m in mem}
        nul = [med(s, top[p]) for p in pool[blk[mem[0]]]]
        R["T3"].append(dict(group=int(g), student_on_member_top=a, member_on_student_neg=b,
                            null_student_on_pool_top_share_pos=float(np.mean(np.array(nul) > 0)),
                            split=bool(min(a.values()) < 0 or max(b.values()) > 0)))
    # ---- T2 ----
    for g in groups:
        mem = np.where((grp == g) & (role != "student"))[0]
        for i, j in itertools.permutations(mem, 2):
            if blk[i] != blk[j]:
                continue
            x = med(i, top[j])
            nul = np.array([med(i, top[p]) for p in pool[blk[j]]])
            R["T2"].append(dict(group=int(g), i=str(role[i]), j=str(role[j]), x=x, null_p10=float(np.percentile(nul, 10)),
                                null_median=float(np.median(nul)), pct=float((nul < x).mean()), gates=bool(x < np.percentile(nul, 10))))
    # ---- T4, T6, gate ----
    tu = list(P["feed_teacher_units"]); F = P["feed_teacher"]; names = P["feeder_names_teacher"]
    ok = np.array([("head" in n or ":mlp" in n) and "bias" not in n for n in names])
    def top10(k):
        f = np.where(ok, F[tu.index(k), 0], -np.inf)
        return set(np.argsort(-f)[:10])
    def shared(ms):
        return float(np.mean([len(top10(a) & top10(b)) for a, b in itertools.combinations(ms, 2)]))
    gm = P["gmax"][fit].mean(0)
    rng = np.random.default_rng(3)
    for g in groups:
        mem = list(np.where((grp == g) & (role != "student"))[0])
        x = shared(mem)
        near = {m: pool[blk[m]][np.argsort(np.abs(gm[pool[blk[m]]] - gm[m]))[:5]] for m in mem}
        nul = []
        for _ in range(1000):
            t = [rng.choice(near[m]) for m in mem]
            if len(set(t)) < len(t):
                continue
            nul.append(shared(t))
        R["T4"].append(dict(group=int(g), shared=x, null_p95=float(np.percentile(nul, 95)), null_mean=float(np.mean(nul)),
                            passes=bool(x > np.percentile(nul, 95)),
                            members_top10=[[str(names[i]) for i in sorted(top10(m))] for m in mem]))
        for lo in mem:
            for hi in mem:
                if blk[hi] == blk[lo] + 1:
                    nm = f"b{blk[lo]}:mlp{neu[lo]}"
                    f = F[tu.index(hi), 0]
                    mlp = np.array([":mlp" in n and "bias" not in n for n in names])
                    order = np.argsort(-np.where(mlp, f, -np.inf))
                    rank = int(np.where(names[order] == nm)[0][0]) + 1
                    R["T6"].append(dict(group=int(g), lower=str(role[lo]), upper=str(role[hi]), rank_among_mlp_feeders=rank,
                                        n_mlp=int(mlp.sum()), contribution=float(f[list(names).index(nm)])))
    for mname in ("teacher", "kd"):
        U = list(P[f"feed_{mname}_units"]); FF = P[f"feed_{mname}"]; nm = P[f"feeder_names_{mname}"]
        okm = np.array([("head" in n or ":mlp" in n) and "bias" not in n for n in nm])
        for k in U:
            if role[k] == "random":
                continue
            fp, fn_ = FF[U.index(k), 0], FF[U.index(k), 1]
            up = np.argsort(-np.where(okm, fp, -np.inf))[:10]
            dn = np.argsort(np.where(okm, fn_, np.inf))[:5]
            R["gate"][f"g{grp[k]}_{role[k]}"] = dict(
                unit=f"{mname} b{blk[k] + 1}:{neu[k]}", up10=[str(nm[i]) for i in up],
                up10_sum_top=float(fp[up].sum()), up10_sum_neg=float(fn_[up].sum()),
                down5_neg=[(str(nm[i]), float(fn_[i])) for i in dn],
                const=float(P[f"feed_{mname}_const"][U.index(k)]))
    json.dump(R, open(os.path.join(OUT, fn.replace("poles", "tree_verbs").replace(".npz", ".json")), "w"), indent=1)
    n3 = sum(r["split"] for r in R["T3"]); n2 = sum(r["gates"] for r in R["T2"]); n4 = sum(r["passes"] for r in R["T4"])
    print(f"T3 groups that split: {n3} of {len(R['T3'])} (prediction: fewer than 3)")
    print(f"T2 pairs that gate: {n2} of {len(R['T2'])} (prediction: at least half)")
    print(f"T4 groups above null p95: {n4} of {len(R['T4'])} (prediction: at least 6)")
    print("T6:", [(r["group"], r["lower"], r["upper"], r["rank_among_mlp_feeders"]) for r in R["T6"]])


if __name__ == "__main__":
    main(*sys.argv[1:])
