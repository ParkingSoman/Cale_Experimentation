"""P0 PLANTED TEST (READING CHECK, uses held photos; labelled): photo-mix calibration of the P0 checks.
FANOUT round 1, P0 Input builder. Mandatory per the critic. It asks: can the check scores SEE a better input at all?

Object (picked by): the check scores of check_input.py (picked by P0 as the readouts under test):
  ResNet-18: var rank median (check_bn statistics vs stored running_var) and inner firing-rate Spearman median.
  CLIP:      score-rank median (floor unit scores vs photos; also its stream / heads / MLP parts) and MLP firing-rate
             Spearman median.
Inputs: 4 mix sets of 256 images: 0, 25, 50, 100% held photos, the rest the SHARED pink images at the same positions
  (cache/input_pink_n256_s1.pt). The photos are the SECOND held photo of each class (held positions 1000-1999), 256
  classes spread evenly over the 1,000, in a fixed shuffled order (seed 0). So they are disjoint from the 1,000
  reference photos (the first held photo of each class) that check_input compares against. Photos keep the network's
  own normalization (no chan_norm); pink keeps its own.
Pass rules (fixed BEFORE the run; no knob is tuned on them):
  (1) each GATING score rises strictly with the mix: s(0%) < s(25%) < s(50%) < s(100%). Gating scores = the proposal's
      three named readouts: ResNet var rank median, ResNet firing-rate Spearman median, CLIP score-rank median. The CLIP
      parts (stream, heads on 12 units, MLP) and the CLIP MLP rate Spearman are printed only (they never gate);
  (2) ResNet var rank median: s(100%) - s(0%) >= .40;
  (3) whole network (ResNet only): the floor (rules/noisecut.py, unchanged) built on the 50% mix at f = .05 has KL on
      1,000 test photos strictly between 7.541 (board: Minitron on photos) and 14.949 (floor on the shared pink file).
  (1) or (2) fails: the check is blind, and its result on self / null is void (the board rows still stand).
Implementation check (passes by algebra): the 0% mix IS the shared pink file, so its scores equal check_input's pink row.
NOT a board row: the mixes contain held photos. Nothing here goes into board.json or cache/. Output: fanout/check_mix.json.

PHOTO USE: held photos, reading only: 1,000 as the check reference, 256 others inside the mixes (this is a reading
check, never a calculation); 1,000 test photos for the KL of rule (3), scoring only.

Usage (from research/clean; ONE torch job at a time):
  curves/.venv/bin/python -I distill/fanout/check_mix.py --net r18
  curves/.venv/bin/python -I distill/fanout/check_mix.py --net clip
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import board  # noqa: E402
import check_input as ci  # noqa: E402
import vitgen  # noqa: E402
from rules import noisecut  # noqa: E402

PINK = os.path.join(HERE, "cache", "input_pink_n256_s1.pt")
OUT = os.path.join(HERE, "check_mix.json")
MIXES = (0.0, 0.25, 0.5, 1.0)
N = 256
KL_PHOTO, KL_PINK = 7.541, 14.949     # ResNet 5%: Minitron on photos (board) / floor on the shared pink file
VAR_GAP = 0.40
SCORES = {"r18": ("var_rank_median", "rate_spearman_median"),          # GATE rule 1 (the proposal's named readouts)
          "clip": ("score_rank_median",)}
INFO = {"r18": ("mean_rank_median", "var_ratio_median"),                 # printed only, never gate
        "clip": ("stream_rank", "head_rank_median", "mlp_rank_median", "rate_spearman_median")}


def pick():
    """Positions inside held[1000:2000] (second held photo per class, class order): 256 classes spread evenly,
    then a fixed shuffle so every mix level holds a class-spread subset."""
    sel = np.round(np.linspace(0, 999, N)).astype(int)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(0)).numpy()
    return sel[perm]


def mixes(photos):
    pink = torch.load(PINK, weights_only=True).float()
    out = {}
    for p in MIXES:
        k = int(round(p * N))
        out[f"mix{int(round(100 * p)):03d}"] = torch.cat([photos[:k].float(), pink[k:]])
    return out, pink


def verdict(net, reps):
    names = [f"mix{int(round(100 * p)):03d}" for p in MIXES]
    res = {}
    for key in SCORES[net]:
        v = [reps[n][key] for n in names]
        res[key] = dict(values=v, rises_strictly=bool(all(b > a for a, b in zip(v, v[1:]))))
    info = {}
    for key in INFO[net]:
        v = [reps[n][key] for n in names]
        info[key] = dict(values=v, rises_strictly=bool(all(b > a for a, b in zip(v, v[1:]))))
    out = dict(rule1_all_rise=all(r["rises_strictly"] for r in res.values()), scores=res, info_not_gating=info)
    if net == "r18":
        gap = res["var_rank_median"]["values"][-1] - res["var_rank_median"]["values"][0]
        out.update(rule2_var_rank_gap=gap, rule2_pass=bool(gap >= VAR_GAP))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=("r18", "clip"), required=True)
    a = ap.parse_args()
    reps = {}
    if a.net == "r18":
        A = board.r18_A().to(board.DEV).eval()
        xh, _ = board.r18_x("held", 2000)
        ref = ci.r18_raw(A, xh[:1000])
        mx, pink = mixes(xh[1000:2000][pick()])
        for name, X in mx.items():
            reps[name] = ci.r18_report(A, ci.r18_raw(A, X), ref)
            ci.r18_print(name, reps[name])
    else:
        T = board.clip_teacher()
        Td = vitgen.to(T, board.DEV)
        XP, _ = ci.clip_photos(0, 1000)
        ref = ci.clip_raw(Td, XP)
        X2, ids2 = ci.clip_photos(1000, 2000)
        mx, pink = mixes(X2[pick()])
        for name, X in mx.items():
            reps[name] = ci.clip_report(ci.clip_raw(Td, X), ref)
            ci.clip_print(name, reps[name])
    v = verdict(a.net, reps)
    v["implementation_check_mix000_is_pink_file"] = bool(torch.equal(mx["mix000"], pink))
    if a.net == "r18":
        S = noisecut.r18(A, 0.05, X=mx["mix050"])     # reading check: this X holds 128 held photos
        res, _ = board.r18_score(S, 1000, "test")
        v["rule3"] = dict(f=0.05, input="mix050 (128 held photos + 128 pink; READING CHECK, not a board row)",
                          kl=res["kl"], agree=res["agree"], top1=res["top1"], bounds=[KL_PHOTO, KL_PINK],
                          pass_=bool(KL_PHOTO < res["kl"] < KL_PINK))
        print(f"rule 3 (whole network, ResNet 5%, floor on mix050): KL {res['kl']:.3f}, agree {100 * res['agree']:.1f}%, "
              f"top-1 {100 * res['top1']:.1f}% | pass if {KL_PHOTO} < KL < {KL_PINK}: {v['rule3']['pass_']}")
    for key, r in v["scores"].items():
        print(f"rule 1 {key}: " + " < ".join(f"{x:.3f}" for x in r["values"]) + f"  rises strictly: {r['rises_strictly']}")
    for key, r in v["info_not_gating"].items():
        print(f"  (info, not a gate) {key}: " + ", ".join(f"{x:.3f}" for x in r["values"]) + f"  rises: {r['rises_strictly']}")
    if a.net == "r18":
        print(f"rule 2 var rank gap (100% - 0%) {v['rule2_var_rank_gap']:.3f} >= {VAR_GAP}: {v['rule2_pass']}")
    blind = not v["rule1_all_rise"] or (a.net == "r18" and not v["rule2_pass"])
    v["check_blind"] = blind
    print(f"PLANTED TEST ({a.net}): {'FAIL - the check is blind; its self/null result is void' if blind else 'PASS - the check sees photo content'}"
          f" | implementation check mix000 == pink file: {v['implementation_check_mix000_is_pink_file']}")
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    d[a.net] = dict(verdict=v, reports=reps, when=datetime.datetime.now().isoformat(timespec="seconds"),
                    photo_use="held photos, reading only (1,000 reference + 256 inside the mixes); 1,000 test photos for rule 3 KL",
                    label="READING CHECK (planted test), not a calculation, not a board row")
    json.dump(d, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} [{a.net}]")


if __name__ == "__main__":
    main()
