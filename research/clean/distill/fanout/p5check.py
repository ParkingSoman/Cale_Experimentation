"""P5 Rank checks (FANOUT round 1). RUNNER ONLY (loads a model; one torch job on the Mac at a time).

Usage (from research/clean):
  curves/.venv/bin/python -I distill/fanout/p5check.py --net r18  --input distill/fanout/cache/input_pink_n256_s1.pt
  curves/.venv/bin/python -I distill/fanout/p5check.py --net clip --input distill/fanout/cache/input_pink_n256_s1.pt

Order inside one call:
  1. SMOKE (seconds): a 2-step greedy on 32 inputs (not saved) for the rule and the null; every builder of
     rules/p5rank*.py runs on it (whole, split sets, one-shot, null) and the built network runs 2 inputs.
  2. IMPLEMENTATION CHECKS (pass by algebra, no evidence):
     I1 masked forward with binary masks = the plain cut, CPU float64, 8 inputs, at f = .95 and .5 (smoke order).
        CLIP: proxy KL on the probe output o = 100 e/||e|| against a fixed random unit class matrix (no photos,
        no class file) + max relative error of e. ResNet: KL of the logits. Pass: KL < 1e-5.
     I2 null hooks with the identity pairing give exactly dJ/dm (rule score), 16 inputs. Pass: rel diff < 1e-4.
     T1 (CLIP) block-6 MLP neurons with pr_w columns set to 0 (picked by noisecut's score: its top 8 in block 6)
        score exactly 0. T2 (ResNet) ReLU rescale (bn1 x 4, conv2 columns / 4) of 32 layer3[0] inner channels
        (picked by p5rank's own step-1 score, top 32): score unchanged, rel diff < 1e-3.
  3. THE ORDER (the long step, minutes): p5rank's nested order on the input, saved to fanout/cache/ (board rows
     reuse it).
  4. PLANTED TEST THAT CAN FAIL (CLIP) T3: stream dims 163, 419, 171, 573, 97 (picked by BOARD F1) all in the top
     5% (top 38 of 768) of the step-1 stream scores. Also printed: their greedy rank.
  5. Per-step probe-error readout (no photos): all parts cut, and each part alone.
Output: printed JSON + fanout/out/p5check_<net>_<tag>.json. Photo use: none.
"""
import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, DIST)
import board  # noqa: E402
import vitgen  # noqa: E402
from rules import p5rank as P  # noqa: E402

LOUD = (163, 419, 171, 573, 97)
OUT = os.path.join(HERE, "out")


def log(*a):
    print("[p5check]", *a, file=sys.stderr, flush=True)


def cast(W, dtype, dev):
    if isinstance(W, dict):
        return {k: cast(v, dtype, dev) for k, v in W.items()}
    if isinstance(W, list):
        return [cast(v, dtype, dev) for v in W]
    return W.to(dev).to(dtype) if torch.is_tensor(W) and W.is_floating_point() else W


def kl(za, zb):
    la, lb = F.log_softmax(za.double(), 1), F.log_softmax(zb.double(), 1)
    return float((la.exp() * (la - lb)).sum(1).mean())


def full_kept(C):
    return {k: torch.ones(len(v), dtype=torch.bool) for k, v in C["step"].items()}


# ---------------------------------------------------------------- 1. smoke
def smoke(net, big, X):
    res = {}
    Cs = P.order(net, big, X[:32], seed=1, sched=(1.0, .95), tag="smoke", save=False)
    Cn = P.order(net, big, X[:32], seed=1, null=True, sched=(1.0, .95), tag="smoke", save=False)
    xs = X[:2]
    for lab, C, which, one in [("whole", Cs, "all", False), ("oneshot", Cs, "all", True), ("null", Cn, "all", False)] + \
            [(f"split_{w}", Cs, w, False) for w in (("stream", "heads", "mlp") if net == "clip" else ("stream", "inner"))]:
        for f in (0.95, 0.5):
            if net == "clip":
                W = P.build_vit(big, f, C, which, one)
                with torch.no_grad():
                    e = vitgen.forward(vitgen.to(W, P.DEV), xs.to(P.DEV)).float().cpu()
                ok = bool(torch.isfinite(e).all()) and tuple(e.shape) == (2, 512)
                shp = vitgen.shape(W)
                res[f"{lab}_f{f:g}"] = dict(ok=ok, d=shp["d"], heads=shp["heads"][0], mlp=shp["mlp"][0])
            else:
                S = P.build_r18(big, f, C, which, one)
                with torch.no_grad():
                    z = S.eval()(xs)
                ok = bool(torch.isfinite(z).all()) and tuple(z.shape) == (2, 1000)
                res[f"{lab}_f{f:g}"] = dict(ok=ok, widths=list(getattr(S, "w", (64, 128, 256, 512))))
    # nesting: 95% kept set contains the 50% kept set, every set
    size = P.sizes_fn(net, big)
    nest = all(set(P.keep(Cs, k, size(0.5)[k]).tolist()) <= set(P.keep(Cs, k, size(0.95)[k]).tolist()) for k in Cs["step"])
    res["nested_95_contains_50"] = nest
    res["pass"] = nest and all(v["ok"] for k, v in res.items() if isinstance(v, dict))
    return res, Cs


# ---------------------------------------------------------------- 2. implementation checks
def i1_masked_eq_cut(net, big, X, Cs):
    out = {}
    xs = X[:8].double()
    size = P.sizes_fn(net, big)
    if net == "clip":
        Td = cast(big, torch.float64, "cpu")
        Cm = F.normalize(torch.randn(512, 1000, generator=torch.Generator().manual_seed(0)).double(), dim=0)
        for f in (1.0, 0.95, 0.5):
            kept = P.kept_at(Cs, size(f))
            M = {k: v.double()[None].expand(len(xs), -1) for k, v in kept.items()}
            with torch.no_grad():
                em = P.vit_fwd(Td, xs, M)
                ec = vitgen.forward(cast(P.build_vit(big, f, Cs), torch.float64, "cpu"), xs)
            om, oc = 100 * F.normalize(em, dim=-1), 100 * F.normalize(ec, dim=-1)
            k = kl(oc @ Cm, om @ Cm)
            rel = float((em - ec).norm() / ec.norm())
            out[f"f{f:g}"] = dict(proxy_kl=k, rel_err_e=rel, ok=k < 1e-5)
    else:
        Am = P.prep_r18(big, "cpu", torch.float64)
        for f in (1.0, 0.95, 0.5):
            kept = P.kept_at(Cs, size(f))
            M = {k: v.double()[None].expand(len(xs), -1) for k, v in kept.items()}
            with torch.no_grad():
                zm = P.r18_out(Am, xs, M)
                zc = P.build_r18(big, f, Cs).double().eval()(xs)
            k = kl(zc, zm)
            out[f"f{f:g}"] = dict(kl=k, max_abs=float((zm - zc).abs().max()), ok=k < 1e-5)
    out["pass"] = all(v["ok"] for v in out.values())
    return out


def i2_identity_null(net, mdl, X, R, Cs, big):
    kept = P.kept_at(Cs, P.sizes_fn(net, big)(0.5))
    ident = {k: torch.arange(len(v)) for k, v in kept.items()}
    S1, _ = P.run_pass(net, mdl, X[:16], R[:16], kept, None, True)
    S2, _ = P.run_pass(net, mdl, X[:16], R[:16], kept, ident, True)
    rel = {k: float((S1[k][kept[k]] - S2[k][kept[k]]).abs().sum() / S1[k][kept[k]].abs().sum().clamp_min(1e-30)) for k in S1}
    worst = max(rel, key=rel.get)
    return dict(max_rel=rel[worst], worst_set=worst, ok=rel[worst] < 1e-4, label="implementation check (passes by algebra)")


def t1_clip(big, X, R):
    from rules import noisecut
    sc = noisecut.vit_scores(big, X=X[:64])["mlp"][6]
    idx = torch.topk(sc, 8).indices
    T1 = dict(big)
    T1["blocks"] = list(big["blocks"])
    B6 = dict(big["blocks"][6])
    B6["pr_w"] = B6["pr_w"].clone()
    B6["pr_w"][:, idx] = 0
    T1["blocks"][6] = B6
    Td1 = vitgen.to(T1, P.DEV)
    full = {k: torch.ones(n, dtype=torch.bool) for k, n in P.sizes_fn("clip", big)(1.0).items()}
    S, _ = P.run_pass("clip", Td1, X[:16], R[:16], full, None, True)
    v = S["mlp6"][idx]
    return dict(object="block-6 MLP neurons, top 8 by noisecut score (picked by noisecut)", idx=idx.tolist(),
                score_max=float(v.max()), block6_median=float(S["mlp6"].median()), ok=float(v.abs().max()) == 0.0,
                label="implementation check (passes by algebra: pr_w = 0 gives g = 0)")


def t2_r18(big, X, R, C):
    idx = torch.topk(C["score1"]["inner20"], 32).indices
    A2 = copy.deepcopy(big).cpu().eval()
    with torch.no_grad():
        bl = A2.layer3[0]
        bl.bn1.weight[idx] *= 4
        bl.bn1.bias[idx] *= 4
        bl.conv2.weight[:, idx] /= 4
    full = {k: torch.ones(n, dtype=torch.bool) for k, n in P.sizes_fn("r18", big)(1.0).items()}
    S1, z1 = P.run_pass("r18", P.prep_r18(big), X[:32], R[:32], full, None, True)
    S2, z2 = P.run_pass("r18", P.prep_r18(A2), X[:32], R[:32], full, None, True)
    a, b = S1["inner20"][idx], S2["inner20"][idx]
    rel = float(((a - b).abs() / a.abs().clamp_min(1e-30)).max())
    return dict(object="layer3[0] inner channels, top 32 by p5rank step-1 score (picked by p5rank)", rel_diff_max=rel,
                logit_max_abs=float((z1 - z2).abs().max()), ok=rel < 1e-3,
                label="implementation check (passes by algebra: a*g is invariant to a ReLU rescale)")


# ---------------------------------------------------------------- 4. T3
def t3_clip(C):
    s = C["score1"]["stream"]
    top = max(1, round(0.05 * len(s)))
    r_greedy = P.rank(C, "stream")
    rows = []
    for u in LOUD:
        rows.append(dict(dim=u, step1_rank_from_top=int((s > s[u]).sum()), step1_score=float(s[u]),
                         greedy_rank_from_top=int(len(s) - 1 - r_greedy[u]),
                         removed_after_step=int(C["step"]["stream"][u]),
                         first_absent_at_f=(C["sched"][int(C["step"]["stream"][u]) + 1]
                                            if int(C["step"]["stream"][u]) < len(C["sched"]) - 1 else "kept at the last f")))
    ok = all(r["step1_rank_from_top"] < top for r in rows)
    return dict(object="stream dims 163, 419, 171, 573, 97 (picked by BOARD F1)", top_k=top, dims=rows, ok=ok,
                stream_score_median=float(s.median()), label="planted test that can fail")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=("clip", "r18"), required=True)
    ap.add_argument("--input", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smoke-only", action="store_true")
    a = ap.parse_args()
    X = torch.load(a.input, weights_only=True).float() if a.input else P.default_input()
    big = board.clip_teacher() if a.net == "clip" else board.r18_A()
    tag = P.input_tag(X if a.input else None)
    out = dict(net=a.net, input=os.path.basename(a.input) or "pink-default", tag=tag, seed=a.seed, photo_use="none")
    t0 = time.time()
    out["smoke"], Cs = smoke(a.net, big, X)
    log("smoke", json.dumps(out["smoke"]))
    out["I1_masked_eq_cut"] = i1_masked_eq_cut(a.net, big, X, Cs)
    log("I1", json.dumps(out["I1_masked_eq_cut"]))
    if a.net == "clip":
        mdl = vitgen.to(big, P.DEV)
        R = P.probe("clip", len(X), a.seed, big["proj"].shape[1])
    else:
        mdl = P.prep_r18(big)
        R = P.probe("r18", len(X), a.seed, big.fc.weight.shape[0])
    out["I2_identity_null"] = i2_identity_null(a.net, mdl, X, R, Cs, big)
    log("I2", json.dumps(out["I2_identity_null"]))
    if a.net == "clip":
        out["T1"] = t1_clip(big, X, R)
        log("T1", json.dumps(out["T1"]))
    del mdl
    out["smoke_and_checks_seconds"] = round(time.time() - t0, 1)
    pre_ok = bool(out["smoke"]["pass"] and out["I1_masked_eq_cut"]["pass"] and out["I2_identity_null"]["ok"]
                  and out.get("T1", {"ok": True})["ok"])
    if not pre_ok:
        out["implementation_checks_pass"] = False
        os.makedirs(OUT, exist_ok=True)
        p = os.path.join(OUT, f"p5check_{a.net}_{tag}_s{a.seed}.json")
        json.dump(out, open(p, "w"), indent=1)
        print(json.dumps(out, indent=1))
        print("IMPLEMENTATION CHECK FAILED: the order was NOT built and nothing was cached. Saved", p)
        sys.exit(1)
    if not a.smoke_only:
        C = P.order(a.net, big, X if a.input else None, seed=a.seed)
        out["order_seconds"] = C["seconds"]
        if a.net == "clip":
            out["T3"] = t3_clip(C)
        else:
            out["T2"] = t2_r18(big, X, R, C)
        out["readout"] = C["readout"]
    out["implementation_checks_pass"] = bool(out["smoke"]["pass"] and out["I1_masked_eq_cut"]["pass"] and out["I2_identity_null"]["ok"]
                                             and out.get("T1", out.get("T2", {"ok": True}))["ok"])
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"p5check_{a.net}_{tag}_s{a.seed}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "readout"}, indent=1))
    print("readout (probe error, no photos):")
    for r in out.get("readout", []):
        print(json.dumps(r))
    print("saved", p)


if __name__ == "__main__":
    main()
