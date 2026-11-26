"""P0 checks for ANY generic input file (FANOUT round 1, P0 Input builder). READING ONLY: no calculation reads these
numbers, and no knob is tuned on them (tuning uses only board.py --split held KL).

ResNet-18 (the known-answer check, extended; check_bn.py itself is not edited, its functions are imported):
  - check_bn.stats on each input file and on 1,000 held photos (the ceiling, reading only). Per BatchNorm layer: Spearman
    of the measured per-channel input variance (mean) with the stored running_var (running_mean), and the median ratio
    measured / stored variance. The stored BatchNorm statistics are the KNOWN ANSWER only, never data.
  - Per stage (stem, layer1-4): medians of var rank, mean rank, var ratio.
  - Inner firing rate: P(bn1 output > 0) per inner channel of each block (one hook, same forward). Per block: Spearman
    against the photo rates, median rate ratio (channels with a photo rate > 0).
  - Guards (a trip is recorded as a fail): layer4 var ratio median > 2; fraction of inner channels with rate > .9 more
    than 2x the photo fraction.
CLIP (no BatchNorm, so the reference is the same reading on 1,000 held photos, reading only):
  - The floor's (rules/noisecut.py) unit scores on the input against the same scores on the photos: Spearman of the
    stream-dim scores (768), per block of the head scores (12) and of the MLP scores (3,072). score_rank_median =
    median of (stream, median over blocks of heads, median over blocks of MLP).
    The loop is a copy of noisecut.vit_scores (same formulas); the implementation check compares it with
    noisecut.vit_scores itself on the first input (passes by algebra if the copy is faithful).
  - MLP firing rate P(z > 0), z = fc(ln2(x)), all tokens: Spearman against photos per block, median rate ratio.
  - Loud-dim RMS (stream dims 163, 419, 171, 573, 97) per site (embedding + 12 block outputs): ratio input / photos.
  - Top-10 eigenvalue share of the stream covariance (all tokens) per site.
  - Guards: top-1% score share (MLP per block, median; stream) more than 2x the photo share; fraction of MLP units with
    rate > .9 more than 2x the photo fraction.
  --extra MODULE:FUNC adds a teammate's per-unit score, FUNC(W, X) -> {name: tensor or list of tensors} (e.g. P5's
  score(W, x)); each entry is Spearman-ranked against the same score on the photos.

PHOTO USE: 1,000 held photos, reading only (ResNet: board.r18_x("held", 2000)[:1000], the first held photo of each class,
as check_bn; CLIP: eval_fast.teacher_ref("held") ids[:1000], the same photos). Test photos are not read here.

Usage (from research/clean; ONE torch job at a time):
  curves/.venv/bin/python -I distill/fanout/check_input.py --net r18  --inputs distill/fanout/cache/input_pink_n256_s1.pt
  curves/.venv/bin/python -I distill/fanout/check_input.py --net clip --inputs A.pt,B.pt,C.pt
Output: fanout/check_input.json = {net: {input name: report}} (merged; a rerun of a name replaces it).
"""
import argparse
import datetime
import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, DIST)
import board  # noqa: E402
import check_bn  # noqa: E402
import vitgen  # noqa: E402
from rules import noisecut  # noqa: E402

DEV = board.DEV
OUT = os.path.join(HERE, "check_input.json")
PHOTO = "photos_held1000 (reading only)"
LOUD = (163, 419, 171, 573, 97)
spearman = check_bn.spearman


def _med(v):
    v = np.asarray(v, dtype=float)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")


def _ratio_med(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = b > 0
    return float(np.median(a[m] / b[m])) if m.any() else float("nan")


def top_share(s, frac=0.01):
    s = np.sort(np.asarray(s, dtype=float))[::-1]
    k = max(1, int(np.ceil(frac * len(s))))
    return float(s[:k].sum() / s.sum()) if s.sum() > 0 else float("nan")


# ---------------------------------------------------------------- ResNet-18
def r18_raw(A, x):
    """check_bn.stats (BN inputs) plus inner firing rates from one hook on each block's bn1 output, same forward."""
    acc, hooks = {}, []
    for s, layer in enumerate((A.layer1, A.layer2, A.layer3, A.layer4)):
        for b, bl in enumerate(layer):
            def h(mod, inp, out, name=f"layer{s + 1}.{b}"):
                on = (out > 0).float().sum((0, 2, 3))
                c, n = acc.get(name, (0, 0))
                acc[name] = (c + on, n + out.shape[0] * out.shape[2] * out.shape[3])
            hooks.append(bl.bn1.register_forward_hook(h))
    try:
        st = check_bn.stats(A, x)
    finally:
        for hk in hooks:
            hk.remove()
    rates = {k: (c / n).cpu().numpy() for k, (c, n) in acc.items()}
    return st, rates


def _stage(name):
    return name.split(".")[0] if name.startswith("layer") else "stem"


def r18_report(A, raw, ref):
    """raw, ref: outputs of r18_raw for the input and for the photos."""
    st, rates = raw
    _, ref_rates = ref
    rows = []
    for n, m in A.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            mu, var = st[n]
            rv, rm = m.running_var.cpu().numpy(), m.running_mean.cpu().numpy()
            rows.append(dict(layer=n, var_rank=spearman(var, rv), mean_rank=spearman(mu, rm),
                             var_ratio_median=float(np.median(var / rv))))
    stages = {}
    for sname in ("stem", "layer1", "layer2", "layer3", "layer4"):
        rs = [r for r in rows if _stage(r["layer"]) == sname]
        stages[sname] = dict(var_rank=_med([r["var_rank"] for r in rs]), mean_rank=_med([r["mean_rank"] for r in rs]),
                             var_ratio=_med([r["var_ratio_median"] for r in rs]))
    blocks = {k: dict(rate_spearman=spearman(rates[k], ref_rates[k]), rate_ratio_median=_ratio_med(rates[k], ref_rates[k]),
                      rate_mean=float(rates[k].mean())) for k in rates}
    all_r = np.concatenate([rates[k] for k in rates])
    all_ref = np.concatenate([ref_rates[k] for k in ref_rates])
    frac_hi, frac_hi_ph = float((all_r > 0.9).mean()), float((all_ref > 0.9).mean())
    guards = dict(layer4_var_ratio_gt_2=bool(stages["layer4"]["var_ratio"] > 2),
                  rate_gt_09_frac_over_2x_photo=bool(frac_hi > 2 * frac_hi_ph and frac_hi > 0))
    vr = [r["var_rank"] for r in rows]
    return dict(var_rank_median=_med(vr), var_rank_min=float(np.nanmin(vr)),
                mean_rank_median=_med([r["mean_rank"] for r in rows]),
                var_ratio_median=_med([r["var_ratio_median"] for r in rows]),
                rate_spearman_median=_med([b["rate_spearman"] for b in blocks.values()]),
                rate_ratio_median=_med([b["rate_ratio_median"] for b in blocks.values()]),
                frac_rate_gt_09=frac_hi, frac_rate_gt_09_photos=frac_hi_ph,
                guards=guards, any_guard_tripped=any(guards.values()), stages=stages, blocks=blocks, layers=rows)


def r18_photos():
    xp, _ = board.r18_x("held", 2000)
    return xp[:1000]


def r18_print(name, r):
    s = r["stages"]
    print(f"{name:34s} var rank med {r['var_rank_median']:.3f} (min {r['var_rank_min']:.2f}) | mean rank {r['mean_rank_median']:.3f} | "
          f"var ratio {r['var_ratio_median']:.3f} | stage var rank " + " ".join(f"{s[k]['var_rank']:.2f}" for k in s) +
          " | stage var ratio " + " ".join(f"{s[k]['var_ratio']:.2f}" for k in s) +
          f" | rate rho {r['rate_spearman_median']:.3f} ratio {r['rate_ratio_median']:.2f} | rate>.9 {r['frac_rate_gt_09']:.4f}"
          f" (photos {r['frac_rate_gt_09_photos']:.4f}) | guard {'TRIP' if r['any_guard_tripped'] else 'ok'}")


# ---------------------------------------------------------------- CLIP
@torch.no_grad()
def clip_raw(Td, X, bs=noisecut.BS):
    """Copy of noisecut.vit_scores' loop (same formulas, same batch averaging), plus MLP firing rates, loud-dim RMS and the
    stream covariance per site. X: (N, 3, 224, 224) normalized, any dtype, CPU."""
    nb = len(Td["blocks"])
    H = Td["blocks"][0]["q_w"].shape[0] // vitgen.HEAD_DIM
    d = Td["cls"].numel()
    st = torch.zeros(d, device=DEV)
    hd = torch.zeros(nb, H, device=DEV)
    ml, rate = [0.0] * nb, [0.0] * nb
    loud = torch.zeros(nb + 1, len(LOUD), device=DEV)
    s1 = torch.zeros(nb + 1, d, dtype=torch.float64)
    s2 = torch.zeros(nb + 1, d, d, dtype=torch.float64)
    ntok = 0
    li = torch.tensor(LOUD, device=DEV)
    cnt = 0

    def site(j, s):
        t = s.reshape(-1, d)
        s1[j] += t.sum(0).cpu().double()
        s2[j] += (t.T @ t).cpu().double()
        loud[j] += s[..., li].pow(2).mean((0, 1))

    for a in range(0, len(X), bs):
        x = X[a:a + bs].to(DEV).float()
        s = vitgen.embed(Td, x)
        st += s.pow(2).mean((0, 1))
        site(0, s)
        ntok += s.shape[0] * s.shape[1]
        for i, B in enumerate(Td["blocks"]):
            y = vitgen.norm(s, B["ln1"])
            N, L, _ = y.shape
            q = F.linear(y, B["q_w"], B["q_b"]).view(N, L, H, 64).transpose(1, 2)
            k = F.linear(y, B["k_w"], B["k_b"]).view(N, L, H, 64).transpose(1, 2)
            v = F.linear(y, B["v_w"], B["v_b"]).view(N, L, H, 64).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v)
            ow = B["o_w"].view(d, H, 64)
            per_head = torch.einsum("nhlc,dhc->nhld", o, ow)
            hd[i] += per_head.pow(2).mean((0, 2, 3))
            s = s + per_head.sum(1) + B["o_b"]
            z = F.linear(vitgen.norm(s, B["ln2"]), B["fc_w"], B["fc_b"])
            rate[i] = rate[i] + (z > 0).float().mean((0, 1))
            g = F.gelu(z)
            ml[i] = ml[i] + g.abs().mean((0, 1))
            s = s + F.linear(g, B["pr_w"], B["pr_b"])
            st += s.pow(2).mean((0, 1))
            site(i + 1, s)
        cnt += 1
    mu = s1 / ntok
    top10 = []
    for j in range(nb + 1):
        cov = s2[j] / ntok - torch.outer(mu[j], mu[j])
        ev = torch.linalg.eigvalsh(cov).clamp_min(0)
        top10.append(float(ev[-10:].sum() / ev.sum()))
    return dict(stream=(st / cnt).sqrt().cpu(), head=[(hd[i] / cnt).sqrt().cpu() for i in range(nb)],
                mlp=[(ml[i] / cnt).cpu() for i in range(nb)], rate=[(rate[i] / cnt).cpu() for i in range(nb)],
                loud=(loud / cnt).sqrt().cpu(), top10=top10)


def clip_impl_check(T, X, raw):
    """Implementation check (passes by algebra if clip_raw copies noisecut faithfully)."""
    ref = noisecut.vit_scores(T, X=X)
    err = lambda a, b: float((a - b).abs().max() / b.abs().max())  # noqa: E731
    e = max([err(raw["stream"], ref["stream"])] + [err(a, b) for a, b in zip(raw["head"], ref["head"])] +
            [err(a, b) for a, b in zip(raw["mlp"], ref["mlp"])])
    res = dict(max_rel_err_vs_noisecut=e, passed=bool(e < 1e-4))
    print("[implementation check, passes by algebra] clip_raw vs noisecut.vit_scores:", json.dumps(res))
    return res


def clip_report(raw, ref):
    nb = len(raw["mlp"])
    np_ = lambda t: t.numpy()  # noqa: E731
    head = [spearman(np_(raw["head"][i]), np_(ref["head"][i])) for i in range(nb)]
    mlp = [spearman(np_(raw["mlp"][i]), np_(ref["mlp"][i])) for i in range(nb)]
    stream = spearman(np_(raw["stream"]), np_(ref["stream"]))
    rate = [spearman(np_(raw["rate"][i]), np_(ref["rate"][i])) for i in range(nb)]
    rate_ratio = [_ratio_med(np_(raw["rate"][i]), np_(ref["rate"][i])) for i in range(nb)]
    all_r = np.concatenate([np_(r) for r in raw["rate"]])
    all_ref = np.concatenate([np_(r) for r in ref["rate"]])
    frac_hi, frac_hi_ph = float((all_r > 0.9).mean()), float((all_ref > 0.9).mean())
    sh_mlp = _med([top_share(np_(m)) for m in raw["mlp"]])
    sh_mlp_ph = _med([top_share(np_(m)) for m in ref["mlp"]])
    sh_st, sh_st_ph = top_share(np_(raw["stream"])), top_share(np_(ref["stream"]))
    loud_ratio = (raw["loud"] / ref["loud"]).numpy()  # (sites, 5)
    guards = {"mlp_top1pct_share_over_2x_photo": bool(sh_mlp > 2 * sh_mlp_ph),
              "stream_top1pct_share_over_2x_photo": bool(sh_st > 2 * sh_st_ph),
              "mlp_rate_gt_09_frac_over_2x_photo": bool(frac_hi > 2 * frac_hi_ph and frac_hi > 0)}
    return dict(score_rank_median=_med([stream, _med(head), _med(mlp)]), stream_rank=stream,
                head_rank_median=_med(head), mlp_rank_median=_med(mlp), head_rank=head, mlp_rank=mlp,
                rate_spearman_median=_med(rate), rate_ratio_median=_med(rate_ratio), rate_spearman=rate,
                rate_ratio=rate_ratio, frac_rate_gt_09=frac_hi, frac_rate_gt_09_photos=frac_hi_ph,
                loud_dims=list(LOUD), loud_rms=raw["loud"].numpy().tolist(),
                loud_ratio_vs_photos=loud_ratio.tolist(), loud_ratio_vs_photos_median_per_dim=np.median(loud_ratio, 0).tolist(),
                top10_eig_share=raw["top10"], top10_eig_share_photos=ref["top10"],
                top1pct_share=dict(mlp=sh_mlp, mlp_photos=sh_mlp_ph, stream=sh_st, stream_photos=sh_st_ph),
                guards=guards, any_guard_tripped=any(guards.values()))


def clip_photos(lo=0, hi=1000):
    sys.path.insert(0, os.path.join(DIST, "clipkd"))
    import eval_fast
    ref = eval_fast.teacher_ref("held")
    assert abs(ref["scale"] - 100.0) < 1e-2, f"stored logit scale is {ref['scale']}, make_input assumes 100"
    return eval_fast.photos(ref["ids"][lo:hi]), ref["ids"][lo:hi]


def clip_print(name, r):
    print(f"{name:34s} score rank med {r['score_rank_median']:.3f} (stream {r['stream_rank']:.3f}, heads {r['head_rank_median']:.3f}, "
          f"MLP {r['mlp_rank_median']:.3f}) | MLP rate rho {r['rate_spearman_median']:.3f} ratio {r['rate_ratio_median']:.2f} | "
          f"loud ratio/photos " + " ".join(f"{v:.2f}" for v in r["loud_ratio_vs_photos_median_per_dim"]) +
          f" | top10 eig share med {np.median(r['top10_eig_share']):.3f} (photos {np.median(r['top10_eig_share_photos']):.3f}) | "
          f"guard {'TRIP' if r['any_guard_tripped'] else 'ok'}")


# ---------------------------------------------------------------- teammate scores
def extra_scores(spec, W, X):
    mod, fn = spec.split(":")
    out = getattr(importlib.import_module(mod), fn)(W, X)
    return {k: ([t.detach().float().cpu() for t in v] if isinstance(v, (list, tuple)) else [v.detach().float().cpu()])
            for k, v in out.items()}


def extra_report(e, e_ref):
    return {k: _med([spearman(a.numpy().ravel(), b.numpy().ravel()) for a, b in zip(e[k], e_ref[k])]) for k in e}


# ---------------------------------------------------------------- main
def load_input(p):
    X = torch.load(p, weights_only=True).float()   # as board.py loads --input
    assert X.dim() == 4 and tuple(X.shape[1:]) == (3, 224, 224), X.shape
    return os.path.basename(p)[:-3], X


def save(net, reports, path=OUT):
    d = json.load(open(path)) if os.path.exists(path) else {}
    d.setdefault(net, {}).update(reports)
    json.dump(d, open(path, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=("r18", "clip"), required=True)
    ap.add_argument("--inputs", required=True, help="comma-separated .pt files of normalized images")
    ap.add_argument("--extra", default="", help="MODULE:FUNC, a teammate's score(W, X) -> {name: per-unit tensors}")
    a = ap.parse_args()
    inputs = [load_input(p) for p in a.inputs.split(",") if p]
    when = datetime.datetime.now().isoformat(timespec="seconds")
    reports = {}
    if a.net == "r18":
        A = board.r18_A().to(DEV).eval()
        ref = r18_raw(A, r18_photos())
        e_ref = extra_scores(a.extra, A, r18_photos()) if a.extra else None
        reports[PHOTO] = r18_report(A, ref, ref)
        r18_print(PHOTO, reports[PHOTO])
        for name, X in inputs:
            reports[name] = r18_report(A, r18_raw(A, X), ref)
            if a.extra:
                reports[name]["extra_rank"] = extra_report(extra_scores(a.extra, A, X), e_ref)
            r18_print(name, reports[name])
    else:
        T = board.clip_teacher()
        Td = vitgen.to(T, DEV)
        XP, _ = clip_photos()
        ref = clip_raw(Td, XP)
        e_ref = extra_scores(a.extra, T, XP) if a.extra else None
        reports[PHOTO] = clip_report(ref, ref)
        clip_print(PHOTO, reports[PHOTO])
        for j, (name, X) in enumerate(inputs):
            raw = clip_raw(Td, X)
            reports[name] = clip_report(raw, ref)
            if j == 0:
                reports[name]["implementation_check"] = clip_impl_check(T, X, raw)
            if a.extra:
                reports[name]["extra_rank"] = extra_report(extra_scores(a.extra, T, X), e_ref)
            clip_print(name, reports[name])
    for r in reports.values():
        r.update(when=when, photo_use="1,000 held photos, reading only (reference); no calculation reads this file")
    save(a.net, reports)
    print(f"wrote {OUT} [{a.net}]: " + ", ".join(reports))


if __name__ == "__main__":
    main()
