# scale: neurons=every node of CLIP ViT-B/16 and ResNet-18; net=both; photos=200 test photos, scoring only; where=mac
# rung: Q=tree loop checks (identity + planted siblings); EST=4 min; PERMIT=Shonu approved plan 2026-11-26 late night
"""Checks before any score is read (PAGE.md, plan step 3). Photos: 200 test photos (quick set), scoring only.

identity  f = 1: every group is one node, the build must return the big network (KL < 1e-5).
planted   in a copy of the big network, the PLANT weakest nodes of every block are split into two exact sibling copies
          (same read, write halved). The copy computes the same function. The build at the original sizes must merge
          each pair back (KL < 1e-4). A failure is a code bug, not a concept result.
Usage: curves/.venv/bin/python -I distill/tree/checks.py clip|r18
"""
import copy
import json
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DIST = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(DIST, "fanout"))
import board  # noqa: E402
import build as TB  # noqa: E402

PLANT = 16


def pick_twins(strength):
    """weakest nodes such that every copy (half strength) stays weaker than any merged pair and any other node."""
    order = strength.argsort().tolist()
    s = strength[order]
    m = 1
    while m < min(PLANT, len(order) - 1) and s[m] < 2 * s[0] and s[m] / 2 < s[0]:
        m += 1
    return order[:m]


def plant_clip(T, tr):
    P = copy.deepcopy(T)
    ks = {}
    for b, B in enumerate(P["blocks"]):
        tw = pick_twins(tr["strength"][("m", b)])
        ks[("m", b)] = B["fc_w"].shape[0]
        ks[("h", b)] = B["q_w"].shape[0] // TB.HD
        B["pr_w"] = B["pr_w"].clone()
        B["pr_w"][:, tw] *= 0.5
        B["fc_w"] = torch.cat([B["fc_w"], B["fc_w"][tw]])
        B["fc_b"] = torch.cat([B["fc_b"], B["fc_b"][tw]])
        B["pr_w"] = torch.cat([B["pr_w"], B["pr_w"][:, tw]], 1)
    return P, ks


def plant_r18(A, tr):
    P = copy.deepcopy(A).cpu().eval()
    ks = {}
    for b, blk in enumerate(TB.r18_blocks(P)):
        tw = pick_twins(tr["strength"][("m", b)])
        n = blk.conv1.out_channels
        ks[("m", b)] = n
        W1, W2 = blk.conv1.weight.data, blk.conv2.weight.data.clone()
        W2[:, tw] *= 0.5
        c1 = nn.Conv2d(W1.shape[1], n + len(tw), 3, blk.conv1.stride, 1, bias=False)
        c1.weight.data = torch.cat([W1, W1[tw]])
        c2 = nn.Conv2d(n + len(tw), W2.shape[0], 3, 1, 1, bias=False)
        c2.weight.data = torch.cat([W2, W2[:, tw]], 1)
        b1 = nn.BatchNorm2d(n + len(tw))
        for name in ("weight", "bias", "running_mean", "running_var"):
            t = getattr(blk.bn1, name).data
            getattr(b1, name).data = torch.cat([t, t[tw]])
        b1.eps = blk.bn1.eps
        blk.conv1, blk.bn1, blk.conv2 = c1, b1, c2
    return P.eval(), ks


def main():
    net = sys.argv[1]
    X = torch.load(os.path.join(DIST, "fanout", "cache", f"{net}_self_n256_k32_s1.pt"), weights_only=True).float()
    score = (lambda W: board.clip_score(W, 1000, "test", stride=5)[0]) if net == "clip" else \
        (lambda S: board.r18_score(S, 1000, "test", stride=5)[0])
    big = board.clip_teacher() if net == "clip" else board.r18_A().cpu()
    stats, tree, build = (TB.clip_stats, TB.clip_tree, TB.build_clip) if net == "clip" else (TB.r18_stats, TB.r18_tree, TB.build_r18)
    out = {}
    st = stats(big, X)
    tr = tree(big, st)
    W = build(big, 1.0, X, st=st, tr=tr)
    if net == "clip":
        W.pop("_infos")
    out["identity"] = score(W)
    print("identity", out["identity"], flush=True)
    del W
    P, ks = (plant_clip if net == "clip" else plant_r18)(big, tr)
    out["planted_copy_vs_big"] = score(P)
    print("planted copy (no build) vs big", out["planted_copy_vs_big"], flush=True)
    stP = stats(P, X)
    trP = tree(P, stP)
    nt = {}
    for key, k in ks.items():
        n = trP["strength"][key].numel()
        nt[f"{key[0]}{key[1]}"] = n - k
    Wp = build(P, 1.0, X, st=stP, tr=trP, ks=ks)
    if net == "clip":
        infos = Wp.pop("_infos")
    else:
        infos = Wp._infos
    out["planted_build"] = score(Wp)
    out["twins_per_block"] = nt
    out["groups"] = [dict(block=i["block"], kind=i["kind"], groups=i.get("groups"), max_group=i.get("max_group", 1)) for i in infos]
    print("planted build vs big", out["planted_build"], "twins", nt, flush=True)
    out["pass"] = dict(identity=out["identity"]["kl"] < 1e-5, planted=out["planted_build"]["kl"] < 1e-4)
    json.dump(out, open(os.path.join(HERE, "out", f"checks_{net}.json"), "w"), indent=1)
    print("PASS" if all(out["pass"].values()) else "FAIL", out["pass"])


if __name__ == "__main__":
    main()
