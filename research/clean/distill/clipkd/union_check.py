"""Union check, steps 2-4 (2026-11-25, plan "reconcile the tree checks with the essay's coarse neuron").

Is a student neuron's strong-firing set the UNION of a few teacher neurons' strong-firing sets, better than any
single teacher neuron's set? (essay line 251: "one 'curves & loops' neuron instead of ten curve detectors")

Unit firing = max over the 197 tokens (strongest crop). Strong set = top 1% of photos within a half (same size for
every unit). Union of teacher group = per photo, max over members of (firing / member's own FIT 99th percentile).
For each student neuron: greedy members (up to 4) from teacher blocks b and b-1, chosen on FIT by strong-set overlap
(tie-break: mean union value on the student's FIT strong photos). Step 1 searches all 6,144 teacher neurons; steps
2-4 search the 64 best by the same score. Overlap measured on SCORE for k = 1..4.
Photos for reading only. Output: out/union_check[_smoke].json.
Usage: curves/.venv/bin/python -I distill/clipkd/union_check.py [--smoke]
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
KMAX, NC = 4, 64


def top_mask(A, q):
    """A (n, m): boolean mask of each column's top q rows."""
    idx = np.argpartition(-A, q - 1, axis=0)[:q]
    M = np.zeros(A.shape, bool)
    np.put_along_axis(M, idx, True, axis=0)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    tag = "_smoke" if a.smoke else ""
    rng = np.random.default_rng(0)
    L = {n: np.load(os.path.join(OUT, f"maxfire_{n}{tag}.npz")) for n in ("teacher", "kd", "baseline")}
    for n in L:
        assert (L[n]["photo_ids"] == L["teacher"]["photo_ids"]).all()
    pos = L["teacher"]["pos"]
    fit, sco = pos < 4, pos >= 4
    q = max(2, int(round(.01 * fit.sum())))
    Tf = L["teacher"]["fire"].astype(np.float32)
    nb = Tf.shape[1]
    res = {"q_per_half": q, "photos": int(len(pos)), "sanity": {}, "blocks": {}}

    cls = os.path.join(OUT, f"cls_teacher{tag}.npz")
    if os.path.exists(cls):
        c = np.load(cls)
        same = (c["photo_ids"] == L["teacher"]["photo_ids"][pos < 4]).all()
        res["sanity"]["cls_ids_match"] = bool(same)
        if same:
            d = Tf[pos < 4] - c["fire"].astype(np.float32)
            res["sanity"]["min_maxfire_minus_cls"] = float(d.min())

    def teacher_block(b):
        F = Tf[:, b]
        scale = np.percentile(F[fit], 99, axis=0) + 1e-6
        return F / scale

    def greedy(st_f, st_s, P_f, P_s):
        """st_*: student firing (n,), P_*: teacher normalized firing (n, m). Returns SCORE overlap for k=1..KMAX, members."""
        sm_f = top_mask(st_f[:, None], q)[:, 0]
        sm_s = top_mask(st_s[:, None], q)[:, 0]
        tm_f = TM_F                                         # precomputed top masks of P_f
        ov1 = (tm_f & sm_f[:, None]).sum(0) + 1e-3 * P_f[sm_f].mean(0)
        cand = np.argsort(-ov1)[:NC]
        members = [int(cand[0])]
        u_f, u_s = P_f[:, cand[0]].copy(), P_s[:, cand[0]].copy()
        out = [float((top_mask(u_s[:, None], q)[:, 0] & sm_s).sum() / q)]
        for _ in range(KMAX - 1):
            rest = [c_ for c_ in cand if c_ not in members]
            U = np.maximum(u_f[:, None], P_f[:, rest])
            sc = (top_mask(U, q) & sm_f[:, None]).sum(0) + 1e-3 * U[sm_f].mean(0)
            pick = rest[int(np.argmax(sc))]
            members.append(int(pick))
            u_f, u_s = np.maximum(u_f, P_f[:, pick]), np.maximum(u_s, P_s[:, pick])
            out.append(float((top_mask(u_s[:, None], q)[:, 0] & sm_s).sum() / q))
        return out, members

    norm = {b: teacher_block(b) for b in range(nb)}
    for b in range(1, nb):
        P = np.concatenate([norm[b], norm[b - 1]], 1)
        P_f, P_s = P[fit], P[sco]
        TM_F = top_mask(P_f, q)
        row = {}
        # U0: planted union of 3 teacher neurons (+ noise at 25% of its spread)
        if b >= 3:
            rec = []
            for _ in range(25 if not a.smoke else 5):
                g = rng.choice(P.shape[1], 3, replace=False)
                u = P[:, g].max(1)
                fake = u + .25 * u.std() * rng.standard_normal(len(u))
                _, mem = greedy(fake[fit], fake[sco], P_f, P_s)
                rec.append(len(set(mem[:3]) & set(g.tolist())) >= 2)
            row["U0_recovered"] = float(np.mean(rec))
        for n in ("kd", "baseline"):
            S = L[n]["fire"][:, b].astype(np.float32)
            curves, shuf = [], []
            for j in range(S.shape[1]):
                o, _ = greedy(S[fit, j], S[sco, j], P_f, P_s)
                curves.append(o)
                if j < 100:
                    perm = rng.permutation(sco.sum())
                    o2, _ = greedy(S[fit, j], S[sco, j][perm], P_f, P_s)
                    shuf.append(o2[2])
            C = np.array(curves)
            for k in range(KMAX):
                row[f"{n}_k{k + 1}"] = float(np.median(C[:, k]))
                row[f"{n}_k{k + 1}_mean"] = float(C[:, k].mean())
            row[f"{n}_k3_minus_k1_median_of_diff"] = float(np.median(C[:, 2] - C[:, 0]))
            row[f"{n}_shuffled_k3"] = float(np.mean(shuf))
        res["blocks"][b + 1] = row
        print(b + 1, {k: round(v, 3) for k, v in row.items() if not k.endswith("_mean")}, flush=True)

    with open(os.path.join(OUT, f"union_check{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
