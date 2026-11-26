# scale: neurons=s3:69, A:children, A:all-units; net=A, R, R4; photos=saved; where=mac
"""Rung 1 + 2 verdicts and the unit ledger (plan "Delete set D", 2026-11-24). Mac numpy on saved files only.

Inputs (pulled from the cluster into results/oc_retrain/): oc_retrain.json, identity.npz, oc_respond.json, oc_respond.npz.
Outputs: results/oc_retrain/oc_ledger.csv (one row per unit, C columns empty), oc_ledger_read.json, vecread cards (md).
Usage: curves/.venv/bin/python oc_ledger_read.py [dir]
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "oc_retrain")
sys.path.insert(0, os.path.join(HERE, "..", "..", "vecread"))
P = 69
REF = dict(strong=.758, test=.564)                       # read + write trained single parent (LOG "Rung 2")


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def share_in(v, B):
    """Share of |v|^2 inside the span of the columns of B."""
    q, _ = np.linalg.qr(B)
    p = q @ (q.T @ v)
    return float(p @ p / (v @ v))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main():
    rt = json.load(open(os.path.join(DIR, "oc_retrain.json")))
    idn = np.load(os.path.join(DIR, "identity.npz"))
    rs = json.load(open(os.path.join(DIR, "oc_respond.json")))
    d = np.load(os.path.join(DIR, "oc_respond.npz"))
    out = {}

    # ---------------- rung 1
    runs = rt["runs"]
    lr = rt["chosen"]["R"]
    key = lambda name, s: f"{name}_lr{lr:g}_s{s}" if name == "R" else f"R4_lr{rt['chosen']['R4']:g}_s0"
    r0, r1, r4 = runs[key("R", 0)], runs[key("R", 1)], runs[key("R4", 0)]
    out["sweep"] = {k: dict(lr=v["lr"], held=v["held_kl"], step=v["best_step"], strong=v["rec_strong"], test=v["rec_test"])
                    for k, v in runs.items()}
    out["Z"] = rt["Z"]
    low = {tag: min(dd.items(), key=lambda kv: kv[1]["median"]) for tag, dd in rt["identity"].items()}
    out["R1a"] = {tag: dict(layer=k, median=v["median"], pass_=v["median"] >= .9) for tag, (k, v) in low.items()}
    out["R1b"] = dict(R_s0=(r0["rec_strong"], r0["rec_test"]), R_s1=(r1["rec_strong"], r1["rec_test"]),
                      R4=(r4["rec_strong"], r4["rec_test"]), ref=REF,
                      pass_=bool(min(r0["rec_strong"], r1["rec_strong"]) > REF["strong"]
                                 and min(r0["rec_test"], r1["rec_test"]) > REF["test"]))
    out["R1c"] = dict(whole=rt["agree"]["whole"], pass_=rt["agree"]["whole"] >= .5)

    # ---------------- rung 2
    sh = {t: rs["runs"][t]["share"] for t in rs["runs"]}
    out["path_check"] = {t: abs(v["path_sum"] - v["want"]) / abs(v["want"]) for t, v in rs["runs"].items()}
    out["R2a"] = dict(shares=sh, pass_=bool(sh["R_s0"]["P"] >= .5 and sh["R_s1"]["P"] >= .5))
    sP = {t: d[f"{t}__send__s3b1__P"] for t in ("R_s0", "R_s1")}
    top10 = {t: float(np.sort(v)[::-1][:10].sum() / v.sum()) for t, v in sP.items()}
    out["R2b"] = dict(top10_share_of_P=top10, pass_=bool(min(top10.values()) >= .5))
    inv = d["inv_r1"]
    rng = np.random.default_rng(0)
    null = np.array([spearman(sP["R_s0"], rng.permutation(inv)) for _ in range(200)])
    rho = {t: spearman(v, inv) for t, v in sP.items()}
    out["R2c"] = dict(rho=rho, null99=float(np.percentile(null, 99)),
                      pass_=bool(min(rho.values()) > np.percentile(null, 99)))
    out["sender_order_agree"] = spearman(sP["R_s0"], sP["R_s1"])
    cw = d["calc_write_69"]
    if d["effw_mean__Z"].shape[0] > P:                                       # smoke runs hold only 32 channels
        dR = {t: d[f"effw_mean__{t}"][P] - d["effw_mean__Z"][P] for t in ("R_s0", "R_s1")}
        out["R2d"] = dict(cos={t: cos(v, cw) for t, v in dR.items()},
                          pass_=bool(min(cos(v, cw) for v in dR.values()) >= .5))
    else:
        out["R2d"] = "channel 69 not in this file (smoke)"
    out["swap_sign_agree"] = rs["swap_sign_agree"]

    # per-photo spread of the parent's write change, and share inside the children's paths
    keep = list(d["effw_keep"])
    B = d["child_writes"]                                                    # 512 x 11
    spread = {}
    for t in ("R_s0", "R_s1"):
        dp = d[f"effw_photo__{t}"].astype(float) - d["effw_photo__Z"].astype(float)   # photos x keep x 512
        m = dp.mean(0)
        spread[t] = {int(c): dict(spread=float(((dp[:, j] - m[j]) ** 2).sum(1).mean() / (m[j] @ m[j] + 1e-30)),
                                  in_children=share_in(m[j], B), norm=float(np.linalg.norm(m[j])))
                     for j, c in enumerate(keep)}
    out["sender_writes"] = spread
    out["calc_69"] = dict(in_children=share_in(cw, B), norm=float(np.linalg.norm(cw)))

    # ---------------- vecread
    try:
        import vecread as vr
        dw, dr = vr.Dictionary("s4b0"), vr.Dictionary("s3k")
        top = np.argsort(-sP["R_s0"])[:10].tolist()
        cards = ["# vecread readings (R_s0 - Z)\n"]
        cards.append(vr.card_md("children's summed write (deleted)", vr.read_vector(dw, B.sum(1), "write")))
        cards.append(vr.card_md("single-parent calculated write of s3:69", vr.read_vector(dw, cw, "write")))
        for c in sorted(c for c in set(top + [P]) if c < d["effw_mean__Z"].shape[0]):
            v = d["effw_mean__R_s0"][c] - d["effw_mean__Z"][c]
            cards.append(vr.card_md(f"s3:{c} effective-write change (path share {sP['R_s0'][c]:.3f})",
                                    vr.read_vector(dw, v, "write")))
        rd = d["read__R_s0"].astype(float) - d["read__Z"].astype(float)
        rn = np.abs(d["R_s0__recv__s4b0n"])
        for j in np.argsort(-rn)[:10]:
            cards.append(vr.card_md(f"block-0 n{j} read change (path share {d['R_s0__recv__s4b0n'][j]:.3f})",
                                    vr.read_vector(dr, rd[j].reshape(-1), "read")))
        open(os.path.join(DIR, "oc_vecread_cards.md"), "w").write("\n\n".join(cards))
    except Exception as e:                                                     # reported, never hidden
        out["vecread_error"] = repr(e)

    # ---------------- ledger
    rows = []
    recv = {k.split("__")[2]: (d[k], d.get(k.replace("R_s0", "R_s1"))) for k in d.files if k.startswith("R_s0__recv__")}
    for layer, (v0, v1) in recv.items():
        ik = None if layer == "out" else (f"{layer[:-1]}_inner" if layer.endswith("n") else
                                          ("s0b0_h" if layer == "s0" else f"{layer}_h"))
        ic = idn[f"R_s0__{ik}"] if ik and f"R_s0__{ik}" in idn.files else None
        for j in range(len(v0)):
            row = dict(layer=layer, unit=j, recv_share_s0=float(v0[j]), recv_share_s1=float(v1[j]) if v1 is not None else "",
                       ident_corr=float(ic[j]) if ic is not None else "")
            if layer == "s3b1":
                row.update(send_P_s0=float(sP["R_s0"][j]), send_P_s1=float(sP["R_s1"][j]), inv_r1=float(inv[j]))
                if j < d["effw_mean__Z"].shape[0]:
                    w = d["effw_mean__R_s0"][j] - d["effw_mean__Z"][j]
                    row.update(write_change_norm=float(np.linalg.norm(w)), write_in_children=share_in(w, B))
            if layer == "s4b0n":
                rz, rr = d["read__Z"][j].astype(float), d["read__R_s0"][j].astype(float)
                wz, wr = d["write__Z"][:, j], d["write__R_s0"][:, j]
                row.update(read_change=float(np.linalg.norm(rr - rz) / (np.linalg.norm(rz) + 1e-30)),
                           write_change=float(np.linalg.norm(wr - wz) / (np.linalg.norm(wz) + 1e-30)))
            row.update(C_change_cos="", C_share_in_children="")             # filled when C exists
            rows.append(row)
    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("layer", "unit"), k))
    with open(os.path.join(DIR, "oc_ledger.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    out["ledger_rows"] = len(rows)
    json.dump(out, open(os.path.join(DIR, "oc_ledger_read.json"), "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("R1a", "R1b", "R1c", "path_check", "R2a", "R2b", "R2c", "R2d",
                                         "swap_sign_agree", "sender_order_agree", "calc_69", "ledger_rows")},
                     indent=1, default=float))
    print("parent s3:69 write change:", {t: v.get(P) for t, v in spread.items()})
    if "vecread_error" in out:
        print("vecread error:", out["vecread_error"])


if __name__ == "__main__":
    main()
