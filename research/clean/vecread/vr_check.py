"""Verification of the vector reader (bars from the plan of 2026-11-23, written before any run; tool v2 after the
first run: twin merge, support minimums, read patterns, s3k patch space - see README "History").

K1 entries are real: an entry direction (build photos) separates its held-out cells from same-source other cells
   above the 99th percentile of 100 label shuffles (photos permuted within source). Passing -> kept_<space>.json.
K2 stable: the build direction finds itself (top 1) among held-out directions, >= 80% of kept.
K3 write mixes decode: .6 d_a + .4 d_b -> top 2 of the 5-term fit are a, b in order, >= 80%; noise at half the norm,
   >= 60%. Raw and whitened metric; the better one is the write metric.
K4 reads have a known answer: a probe for entry c trained on held half A (logistic: the bar; LDA: reported), turned
   into its pattern and read against the build dictionary, has c on top, >= 80%.
K5 floor: random vectors (write-like and read-like): top cosine and leftover share.
K6 geometry agrees with data: the top-1% held cells of d_c have IoU with c's cells above the shuffled 99th, >= 80%.
K7 structure: an entry's 5 nearest entries share its category more often than shuffled categories.
Usage: python vr_check.py [space ...] [--smoke]
"""
import json
import os
import sys
import time

import numpy as np

import vecread as vr

RNG = np.random.default_rng(0)
NPERM = int(os.environ.get("VR_NPERM", 100))


def within_source_perms(src, n):
    out = []
    for _ in range(n):
        p = np.arange(len(src))
        for s in np.unique(src):
            i = np.nonzero(src == s)[0]
            p[i] = RNG.permutation(i)
        out.append(p)
    return out


def effect(values, p, ce, src, sums, sd):
    vc = values[p, ce]
    eff, w = 0.0, 0
    for s in np.unique(src[p]):
        m = src[p] == s
        S_all, n_all = sums[s]
        eff += m.sum() * (vc[m].mean() - (S_all - vc[m].sum()) / max(n_all - m.sum(), 1))
        w += m.sum()
    return eff / w / sd


def iou(values, p, ce, thr):
    hot = values >= thr
    inter = hot[p, ce].sum()
    return inter / (hot.sum() + len(p) - inter)


def gather(held, space, img, cell):
    """Activation vectors of held cells (img, cell) on the space's grid. s3k: 3 x 3 patches (stride 2, padding 1)."""
    X = held.maps[vr.MAPS[space]]
    G = vr.SPACES[space][1]
    out = np.zeros((len(img), vr.SPACES[space][0]), np.float32)
    order = np.argsort(img)
    ui, start = np.unique(img[order], return_index=True)
    for a in range(0, len(ui), 400):
        u = ui[a:a + 400]
        x = np.asarray(X[u], np.float32)
        if space == "s3k":
            x = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        lo = start[a]
        hi = start[a + 400] if a + 400 < len(ui) else len(order)
        sel = order[lo:hi]
        loc = np.searchsorted(u, img[sel])
        i, j = cell[sel] // G, cell[sel] % G
        if space == "s3k":
            pt = np.stack([x[loc, :, 2 * i + di, 2 * j + dj] for di in range(3) for dj in range(3)], -1)
            out[sel] = pt.reshape(len(sel), -1)          # channel-major, then 3 x 3: matches unfold / kernel order
        else:
            out[sel] = x[loc, :, i, j]
    return out


def held_dirs(held, dic, entries):
    src = held.src
    G = vr.SPACES[dic.space][1]
    T = {s: 0.0 for s in np.unique(src)}
    M = {s: (src == s).sum() * G * G for s in np.unique(src)}
    allc = np.arange(G * G)
    for a in range(0, held.n_img, 500):
        im = np.repeat(np.arange(a, min(held.n_img, a + 500)), G * G)
        ce = np.tile(allc, len(im) // (G * G))
        x = gather(held, dic.space, im, ce)
        for s in T:
            T[s] = T[s] + x[src[im] == s].sum(0)
    out = {}
    for k in entries:
        p, ce = held.cells(dic, k)
        if len(p) < 10:
            continue
        x = gather(held, dic.space, p, ce)
        d, w = 0.0, 0
        for s in np.unique(src[p]):
            m = src[p] == s
            d = d + m.sum() * (x[m].mean(0) - (T[s] - x[m].sum(0)) / (M[s] - m.sum()))
            w += m.sum()
        out[k] = d / w
    return out


def run_space(space, held, tag, report, prev=None):
    t0 = time.time()
    dic = vr.Dictionary(space, tag)
    if prev is not None:                       # reuse K1, K6, K3, K7 (metric-free or already decided)
        return rerun_metric_tests(space, held, dic, prev, report)
    dic.kept[:] = True
    C, G = vr.SPACES[space]
    src = held.src
    perms = within_source_perms(src, NPERM)
    K = len(dic.D)
    sizes = np.array([len(g) for g in dic.groups])
    r = {"labels": int(sizes.sum()), "entries": K, "merged_entries": int((sizes > 1).sum()),
         "largest_group": int(sizes.max()), "rule_of_cat": dic.rule_of_cat}
    k1_eff, k1_pass, k6_pass = np.full(K, np.nan), np.zeros(K, bool), np.zeros(K, bool)
    for a in range(0, K, 64):
        idx = np.arange(a, min(K, a + 64))
        V = held.project(space, dic.D[idx])
        for j, k in enumerate(idx):
            p, ce = held.cells(dic, k)
            if len(p) < 20:
                continue
            vals = V[:, :, j]
            sums = {s: (vals[src == s].sum(), (src == s).sum() * vals.shape[1]) for s in np.unique(src)}
            sd = vals.std()
            e = effect(vals, p, ce, src, sums, sd)
            k1_eff[k] = e
            k1_pass[k] = e > np.percentile([effect(vals, pp[p], ce, src, sums, sd) for pp in perms], 99)
            thr = np.quantile(vals, 0.99)
            k6_pass[k] = iou(vals, p, ce, thr) > np.percentile([iou(vals, pp[p], ce, thr) for pp in perms], 99)
        print(f"  {space}: K1/K6 {idx[-1] + 1}/{K} entries, {time.time() - t0:.0f}s", flush=True)
    tested = ~np.isnan(k1_eff)
    kept = k1_pass
    r.update({"tested": int(tested.sum()), "kept": int(kept.sum()),
              "kept_by_cat": {c: [int(((dic.cat == c) & kept).sum()), int(((dic.cat == c) & tested).sum())]
                              for c in vr.CATS}})
    r["K1"] = {"pass_share_of_tested": round(float(kept.sum() / max(tested.sum(), 1)), 3)}
    json.dump({"kept": [int(x) for x in dic.key[kept]]}, open(os.path.join(vr.RES, f"kept_{space}{tag}.json"), "w"))
    dic.kept = kept
    KK = np.nonzero(kept)[0]
    k6 = k6_pass[kept].mean()
    r["K6"] = {"share": round(float(k6), 3), "pass": bool(k6 >= 0.8)}
    # K2
    r["K2"] = k2(held, dic, KK)
    print(f"  {space}: K2 done {time.time() - t0:.0f}s", flush=True)
    # K3, K5 write
    r["K3"], r["K5_write"] = {}, {}
    for whiten in (False, True):
        D = dic.unit(whiten)
        n = len(D)
        ok, okn = [], []
        for _ in range(1000):
            a_, b_ = RNG.choice(n, 2, replace=False)
            v = 0.6 * D[a_] + 0.4 * D[b_]
            ok.append(vr.omp(v, D)[0][:2] == [a_, b_])
            g = RNG.standard_normal(D.shape[1])
            okn.append(vr.omp(v + g / np.linalg.norm(g) * 0.5 * np.linalg.norm(v), D)[0][:2] == [a_, b_])
        name = "whitened" if whiten else "raw"
        r["K3"][name] = {"clean": round(float(np.mean(ok)), 3), "noise": round(float(np.mean(okn)), 3),
                         "pass": bool(np.mean(ok) >= 0.8 and np.mean(okn) >= 0.6)}
        lefts, tops = [], []
        for _ in range(500):
            g = dic.metric(RNG.standard_normal(C), False)[0] if not whiten else RNG.standard_normal(C)
            g /= np.linalg.norm(g)
            lefts.append(vr.omp(g, D)[2])
            tops.append(np.abs(D @ g).max())
        r["K5_write"][name] = {"leftover_median": round(float(np.median(lefts)), 3),
                               "leftover_5th": round(float(np.percentile(lefts, 5)), 3),
                               "top_abs_cos_median": round(float(np.median(tops)), 3),
                               "top_abs_cos_95th": round(float(np.percentile(tops, 95)), 3)}
    best = max(("raw", "whitened"), key=lambda m: (r["K3"][m]["pass"], r["K3"][m]["noise"]))
    r["write_metric"] = best
    r["K2"]["pass"] = bool(r["K2"][best]["share"] >= 0.8)
    # K7
    Du = dic.unit(False)
    cat = dic.cat[kept]
    S = Du @ Du.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :5]
    real = (cat[nn] == cat[:, None]).mean()
    null = [(c2[nn] == c2[:, None]).mean() for c2 in (RNG.permutation(cat) for _ in range(1000))]
    r["K7"] = {"same_category": round(float(real), 3), "shuffled_99th": round(float(np.percentile(null, 99)), 3),
               "pass": bool(real > np.percentile(null, 99))}
    # K4, K5 read
    r["K4"], r["K5_read"] = k4(held, dic, KK), {}
    global WHITEN
    for WHITEN in (False, True):
        r["K5_read"]["whitened" if WHITEN else "raw"] = k5_read(dic, C)
    r["K4"]["pass"] = r["K4"][r["write_metric"]]["pass"]
    report[space] = r
    print(json.dumps({space: r}, indent=1), flush=True)


WHITEN = False


def k2(held, dic, KK):
    hd = held_dirs(held, dic, KK)
    KH = np.array([k for k in KK if k in hd])
    out = {}
    for whiten in (False, True):
        Db = dic.metric(dic.D[KH], whiten)
        Dh = dic.metric(np.array([hd[k] for k in KH]), whiten)
        Db /= np.linalg.norm(Db, axis=1, keepdims=True)
        Dh /= np.linalg.norm(Dh, axis=1, keepdims=True)
        ok2 = np.argmax(Db @ Dh.T, axis=1) == np.arange(len(KH))
        out["whitened" if whiten else "raw"] = {
            "share": round(float(ok2.mean()), 3), "n": len(KH),
            "by_cat": {c: round(float(ok2[dic.cat[KH] == c].mean()), 2) for c in vr.CATS if (dic.cat[KH] == c).any()}}
    return out


def k5_read(dic, C):
    D = dic.unit(WHITEN)
    tops, lefts = [], []
    for _ in range(300):
        g = dic.metric(dic.pattern(RNG.standard_normal(C)), WHITEN)[0]
        g /= np.linalg.norm(g)
        tops.append(np.abs(D @ g).max())
        lefts.append(vr.omp(g, D)[2])
    m, sd = dic.read_null(WHITEN)
    zt = []
    for _ in range(300):
        g = dic.metric(dic.pattern(RNG.standard_normal(C)), WHITEN)[0]
        g /= np.linalg.norm(g)
        zt.append((((D @ g) - m) / sd).max())
    return {"top_abs_cos_median": round(float(np.median(tops)), 3),
            "top_abs_cos_95th": round(float(np.percentile(tops, 95)), 3),
            "leftover_median": round(float(np.median(lefts)), 3),
            "top_z_median": round(float(np.median(zt)), 2), "top_z_95th": round(float(np.percentile(zt, 95)), 2)}


def rerun_metric_tests(space, held, dic, prev, report):
    r = dict(prev)
    KK = np.nonzero(dic.kept)[0]
    C = dic.C
    m = r["write_metric"]
    r["K2"] = k2(held, dic, KK)
    r["K2"]["pass"] = bool(r["K2"][m]["share"] >= 0.8)
    r["K4"], r["K5_read"] = k4(held, dic, KK), {}
    global WHITEN
    for WHITEN in (False, True):
        r["K5_read"]["whitened" if WHITEN else "raw"] = k5_read(dic, C)
    r["K4"]["pass"] = r["K4"][m]["pass"]
    report[space] = r
    print(json.dumps({space: {k: r[k] for k in ("K2", "K4", "K5_read")}}, indent=1), flush=True)


def k4(held, dic, KK, n_probe=120):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    C = dic.C
    G = vr.SPACES[dic.space][1]
    src = held.src
    half_a = np.zeros(held.n_img, bool)
    half_a[::2] = True
    ia = np.nonzero(half_a)[0]
    im = RNG.choice(ia, 40000)
    ce = RNG.integers(0, G * G, 40000)
    XA = gather(held, dic.space, im, ce).astype(np.float64)
    mu = XA.mean(0)
    sdv = XA.std(0) + 1e-6
    cov = np.cov(XA, rowvar=False)
    Ci = np.linalg.inv(cov + 1e-3 * np.trace(cov) / C * np.eye(C))
    sample = RNG.choice(KK, size=min(n_probe, len(KK)), replace=False)
    hit_l, hit_d, aucs, cs = [], [], [], []
    for k in sample:
        p, cc = held.cells(dic, k)
        a = half_a[p]
        if a.sum() < 20:
            continue
        sel = RNG.choice(np.nonzero(a)[0], size=min(a.sum(), 3000), replace=False)
        xp = gather(held, dic.space, p[sel], cc[sel]).astype(np.float64)
        # negatives: same-source half-A cells that are not entry k
        pool = np.nonzero(half_a & np.isin(src, np.unique(src[p])))[0]
        ni = RNG.choice(pool, 6000)
        nc = RNG.integers(0, G * G, 6000)
        mine = set(zip(p.tolist(), cc.tolist()))
        keep = np.array([(i, c) not in mine for i, c in zip(ni.tolist(), nc.tolist())])
        xn = gather(held, dic.space, ni[keep][:3000], nc[keep][:3000]).astype(np.float64)
        w_lda = Ci @ (xp.mean(0) - mu)
        Z = (np.concatenate([xp, xn]) - mu) / sdv
        y = np.concatenate([np.ones(len(xp)), np.zeros(len(xn))])
        # regularization chosen by 3-fold cross-validation on half A only (first run used C=0.1: train AUC 1.00,
        # new-photo AUC .79 on misses = overfit probes, not valid known answers)
        # folds split by photo: cells of one photo never sit in both a training and a validation fold
        groups = np.concatenate([p[sel], ni[keep][:3000]])
        cv = list(GroupKFold(n_splits=3).split(Z, y, groups))
        clf = LogisticRegressionCV(Cs=[1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1], cv=cv, scoring="roc_auc", max_iter=300,
                                   use_legacy_attributes=True, l1_ratios=(0.0,)).fit(Z, y)
        w_log = clf.coef_[0] / sdv
        aucs.append(clf.scores_[1.0].mean(0).max())
        cs.append(float(clf.C_[0]))
        for w, hits in ((w_log, hit_l), (w_lda, hit_d)):
            hits.append([vr.geometry(dic, dic.pattern(w), whiten=wh, kind="read")["_top"] == k for wh in (False, True)]
                        + [vr.geometry(dic, dic.pattern(w), whiten=True)["_top"] == k])
    hl, hd_ = np.array(hit_l), np.array(hit_d)
    out = {"whitened_no_hub_correction": {"logistic": round(float(hl[:, 2].mean()), 3),
                                          "lda": round(float(hd_[:, 2].mean()), 3)}}
    for i, name in enumerate(("raw", "whitened")):
        out[name] = {"logistic": round(float(hl[:, i].mean()), 3), "lda": round(float(hd_[:, i].mean()), 3),
                     "n": len(hl), "probe_cv_auc_median": round(float(np.median(aucs)), 3),
                     "C_median": float(np.median(cs)),
                     "pass": bool(hl[:, i].mean() >= 0.8)}
    return out


def confirm_space(space, held, report):
    """Confirmation on untouched photos (held set 2): the frozen tool (dictionary, kept list, metric, hub null) is
    tested once. Nothing is changed after this run."""
    t0 = time.time()
    dic = vr.Dictionary(space)                        # kept list frozen from held set 1
    prev = json.load(open(os.path.join(vr.RES, "vr_check.json")))[space]
    m = prev["write_metric"]
    src = held.src
    perms = within_source_perms(src, NPERM)
    KK = np.nonzero(dic.kept)[0]
    k1, k6, tested = [], [], []
    for a in range(0, len(KK), 64):
        idx = KK[a:a + 64]
        V = held.project(space, dic.D[idx])
        for j, k in enumerate(idx):
            p, ce = held.cells(dic, k)
            if len(p) < 20:
                continue
            vals = V[:, :, j]
            sums = {s_: (vals[src == s_].sum(), (src == s_).sum() * vals.shape[1]) for s_ in np.unique(src)}
            sd = vals.std()
            e = effect(vals, p, ce, src, sums, sd)
            k1.append(e > np.percentile([effect(vals, pp[p], ce, src, sums, sd) for pp in perms], 99))
            thr = np.quantile(vals, 0.99)
            k6.append(iou(vals, p, ce, thr) > np.percentile([iou(vals, pp[p], ce, thr) for pp in perms], 99))
        print(f"  confirm {space}: {min(len(KK), a + 64)}/{len(KK)}, {time.time() - t0:.0f}s", flush=True)
    r = {"kept_frozen": int(len(KK)), "tested": len(k1),
         "K1": {"share": round(float(np.mean(k1)), 3)},
         "K6": {"share": round(float(np.mean(k6)), 3), "pass": bool(np.mean(k6) >= 0.8)}}
    r["K2"] = k2(held, dic, KK)
    r["K2"]["pass"] = bool(r["K2"][m]["share"] >= 0.8)
    r["K4"] = k4(held, dic, KK)
    r["K4"]["pass"] = r["K4"][m]["pass"]
    report[space] = r
    print(json.dumps({space: r}, indent=1), flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tag = "_smoke" if "--smoke" in sys.argv else ""
    spaces = args or list(vr.SPACES)
    if "--confirm" in sys.argv:
        held = vr.Held("_h2")
        report = {}
        for s in spaces:
            confirm_space(s, held, report)
            json.dump(report, open(os.path.join(vr.RES, "vr_check_confirm.json"), "w"), indent=1)
        for s, r in report.items():
            print(s, {k: v.get("pass") for k, v in r.items() if isinstance(v, dict) and "pass" in v}, "K1", r["K1"])
        return
    held = vr.Held(tag)
    report = {}
    out = os.path.join(vr.RES, f"vr_check{tag}.json")
    prior = json.load(open(out)) if ("--reuse-k1" in sys.argv and os.path.exists(out)) else {}
    for s in spaces:
        run_space(s, held, tag, report, prior.get(s))
        old = json.load(open(out)) if os.path.exists(out) else {}
        old.update(report)
        json.dump(old, open(out, "w"), indent=1)
    for s, r in report.items():
        print(s, {k: v.get("pass") for k, v in r.items() if isinstance(v, dict) and "pass" in v},
              "write metric:", r["write_metric"])


if __name__ == "__main__":
    main()
