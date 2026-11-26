# scale: neurons=s3:69, A:children, A:tcn; net=A; photos=saved; where=mac
"""Read and write cards for every neuron of the small system (plan 2026-11-23, "same cut, same readers").

Vectors (all in A's spaces, bn folded in):
  parent    s3:69 as a direction of the stage-3 stream     read: s3 (pattern)   write: s3 (raw)
  child i   A W1_0[i] x bn1 scale (256 x 3 x 3)            read: s3k            write: W2_0[:, i] x bn2 scale, 3x3 sum, s4b0
            child i's condition = its read with channel 69 set to 0 (what it adds beyond the parent)
  OCN       read = channel 69 only, 3 x 3 shape = a-weighted mean of the children's channel-69 kernels; write = sum of
            the children's writes
  TCN s     trained in place of the 11 children (onecoarse/code/oc_tcn.py), read / write saved in oc_tcn.npz (if present)
Output: results/system_cards.json, results/system_cards.md (summary table first, then every card).
"""
import json
import os

import numpy as np

import vecread as vr

CLEAN = os.path.dirname(vr.HERE)
R18 = os.path.join(CLEAN, "neuron", "results", "coarse", "r18")
P, K = 69, 11


def bn_scale(bn):
    g, _, _, v = bn
    return g / np.sqrt(v + 1e-5)


def vectors():
    A = np.load(os.path.join(R18, "w_A.npz"))
    cw = np.load(os.path.join(CLEAN, "onecoarse", "results", "oc_children_w.npz"))
    kids = [int(k) for k in cw["order"][:K]]
    a = cw["score"][kids]
    RA = A["W1_0"] * bn_scale(A["bn1_0"])[:, None, None, None]
    WA = A["W2_0"] * bn_scale(A["bn2_0"])[:, None, None, None]
    ocn_r = np.zeros_like(RA[0])
    ocn_r[P] = (a[:, None, None] * RA[kids, P]).sum(0) / a.sum()
    e = np.zeros(256); e[P] = 1.0
    reads = {"parent s3:69": ("s3", e), "OCN": ("s3k", ocn_r)}
    writes = {"parent s3:69": ("s3", e), "OCN": ("s4b0", WA[:, kids].sum((1, 2)).sum(1))}
    for k in kids:
        reads[f"child n{k}"] = ("s3k", RA[k])
        cond = RA[k].copy(); cond[P] = 0
        reads[f"child n{k} condition"] = ("s3k", cond)
        writes[f"child n{k}"] = ("s4b0", WA[:, k].sum((1, 2)))
    f = os.path.join(R18, "oc_tcn.npz")
    if os.path.exists(f):
        t = np.load(f)
        assert np.allclose(t["ocn_read"][0], ocn_r, atol=1e-5), "OCN read differs from oc_tcn.py's"
        for s in (0, 1):
            if f"tcn{s}_read" in t.files:
                reads[f"TCN seed {s}"] = ("s3k", t[f"tcn{s}_read"][0])
                writes[f"TCN seed {s}"] = ("s4b0", t[f"tcn{s}_write"][:, 0].sum((1, 2)))
    return reads, writes, kids


def top3(card):
    return ", ".join(f"{c} {v:+.2f}" for c, _, v in card["up"][:3])


def bot2(card):
    return ", ".join(f"{c} {v:+.2f}" for c, _, v in card["down"][:2])


def main():
    reads, writes, kids = vectors()
    held = vr.Held()
    dics = {s: vr.Dictionary(s) for s in ("s3", "s3k", "s4b0")}
    out = {"children": kids, "read": {}, "write": {}, "compare": {}}
    for name, (sp, v) in reads.items():
        out["read"][name] = vr.read_vector(dics[sp], v.reshape(-1), kind="read", held=held)
    for name, (sp, v) in writes.items():
        out["write"][name] = vr.read_vector(dics[sp], v.reshape(-1), kind="write", held=held)
    for s in (0, 1):
        n = f"TCN seed {s}"
        if n in reads:
            out["compare"][f"read: {n} vs OCN"] = vr.compare(dics["s3k"], reads[n][1].reshape(-1),
                                                             reads["OCN"][1].reshape(-1), "read", held)
            out["compare"][f"write: {n} vs OCN"] = vr.compare(dics["s4b0"], writes[n][1], writes["OCN"][1], "write", held)
    if "TCN seed 0" in reads and "TCN seed 1" in reads:
        out["compare"]["read: TCN s0 vs s1"] = vr.compare(dics["s3k"], reads["TCN seed 0"][1].reshape(-1),
                                                          reads["TCN seed 1"][1].reshape(-1), "read")
        out["compare"]["write: TCN s0 vs s1"] = vr.compare(dics["s4b0"], writes["TCN seed 0"][1],
                                                           writes["TCN seed 1"][1], "write")
    json.dump(out, open(os.path.join(vr.RES, "system_cards.json"), "w"), indent=1, default=str)

    md = ["# Small system: read and write cards", "",
          "Reads: z vs random reads (floor: median 2.6, 95th 3.4). Writes: raw cosine (s4b0 floor: median .13, 95th .16).",
          "A's real block-0 writes: top cosine median .25; real read kernels: top z median 3.8 (README K5b).", "",
          "| Neuron | Read: top 3 up | Read: top 2 down | Read leftover | Write: top 3 up | Write: top 2 down | Write leftover |",
          "|---|---|---|---|---|---|---|"]
    for name in reads:
        r = out["read"][name]
        w = out["write"].get(name)
        md.append(f"| {name} | {top3(r)} | {bot2(r)} | {r['leftover']:.2f} | "
                  + (f"{top3(w)} | {bot2(w)} | {w['leftover']:.2f} |" if w else " | | |"))
    if out["compare"]:
        md += ["", "| Comparison | raw cosine | pattern cosine (reads) |", "|---|---|---|"]
        for k, c in out["compare"].items():
            md.append(f"| {k} | {c['cosine_raw']:.3f} | {c.get('cosine_of_patterns', '')} |")
    md += ["", "## Cards", ""]
    for name in reads:
        md.append(vr.card_md(f"Read: {name}", out["read"][name]))
        if name in out["write"]:
            md.append(vr.card_md(f"Write: {name}", out["write"][name]))
    open(os.path.join(vr.RES, "system_cards.md"), "w").write("\n\n".join(md[:0]) + "\n".join(md))
    print("\n".join(md[:8 + len(reads) + 2 + 4 + len(out['compare'])]))


if __name__ == "__main__":
    main()
