# serves: TU17, TU18 (UNIT-BRANCH.md)
"""The q4 variants of toy_shelter.py: members also read their district's gate (a gate in front)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import toy_shelter as ts  # noqa: E402

if __name__ == "__main__":
    out = {}
    for kind, bg in (("q4", False), ("q4", True), ("pose_q4", False)):
        X, H1, Z2, Wo, yN, w, inb = ts.make(kind, bg)
        S, T = list(range(4)), list(range(4, 8))
        res = {}
        for kS, kT in ((1, 1), (2, 2), (1, 2)):
            dS, _ = ts.fit(H1, Z2, Wo, yN, w, [(S, inb["S"])], [kS])
            dT, _ = ts.fit(H1, Z2, Wo, yN, w, [(T, inb["T"])], [kT])
            dST, P = ts.fit(H1, Z2, Wo, yN, w, [(S, inb["S"]), (T, inb["T"])], [kS, kT])
            eS = ts.kl(yN, ts.build(yN, H1, Z2, Wo, P[:1]), w).item(); eT = ts.kl(yN, ts.build(yN, H1, Z2, Wo, P[1:]), w).item()
            r = dict(DS=min(dS, eS), DT=min(dT, eT), DST=dST); r["I"] = r["DST"] - r["DS"] - r["DT"]
            res[f"{kS}{kT}"] = r
            print(kind, "bg" if bg else "arcs", kS, kT, {k: round(v, 4) for k, v in r.items()}, flush=True)
        out[f"{kind}_{'bg' if bg else 'arcs'}"] = res
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "toy_q4.json"), "w"), indent=1)
