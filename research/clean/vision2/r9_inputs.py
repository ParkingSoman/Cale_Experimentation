"""R9: label-free territory (inputs, not classes). Rules: VISION-LOG.md R9."""
import json
import numpy as np
from convnext import *
from acts import get_acts

X, y = load_images(); names = [n.split(",")[0] for n in class_names()]
amax, _ = get_acts(); ana, _ = split(y); ya = y[ana]; rng = np.random.default_rng(0)


def terr(a, thr, top):
    med = np.median(a, 0); hi = a.max(0) if top == "max" else np.quantile(a, .999, axis=0)
    return (a - med) / (hi - med + 1e-9) >= thr


def nested(kid, par):
    """for each child: has a nested parent, best parent (Jaccard), containment matrix."""
    k, p = kid.astype(np.float32), par.astype(np.float32); inter = k.T @ p; nk, br = k.sum(0), p.sum(0)
    cont = inter / np.maximum(nk[:, None], 1); jac = inter / np.maximum(nk[:, None] + br[None] - inter, 1)
    ok = (cont >= .8) & (br[None] > nk[:, None]); best = np.where(ok.any(1), np.argmax(np.where(ok, jac, -1), 1), -1)
    return ok.any(1), best, cont


# class-level reference (R8)
cm = {b: np.stack([a[ya == c].mean(0) for c in range(1000)]) for b, a in amax.items()}
TC = {}
for b, c in cm.items():
    med = np.median(c, 0); TC[b] = (c - med) / (c.max(0) - med + 1e-9) >= .5
out = {}
for thr, top in ((.5, "max"), (.3, "max"), (.5, "q999")):
    key = f"{thr}_{top}"; T = {b: terr(a, thr, top) for b, a in amax.items()}; o = {"scale": {}}
    print(f"\n== territory threshold {thr}, top = {top}")
    for b in range(12, 18):
        has, best, cont = nested(T[b], T[b - 1]); hs, _, _ = nested(T[b], T[b - 1][rng.permutation(len(ana))])
        o["scale"][b] = {"median_breadth": float(np.median(T[b].sum(0))), "has": float(has.mean()), "null": float(hs.mean())}
        print(f"block {b} <- {b-1}: median breadth {np.median(T[b].sum(0)):.0f} images | nested parent {has.mean():.2f} | shuffled null {hs.mean():.2f}")
        if b == 17:
            hC, bC, cC = nested(TC[17], TC[16]); both = has & hC
            r = np.corrcoef(cont.ravel(), cC.ravel())[0, 1]; same = float((best[both] == bC[both]).mean()) if both.any() else 0
            o["agree"] = {"corr_containment": float(r), "same_top_parent": same, "n_both": int(both.sum())}
            print(f"  agreement with class level (16->17): containment corr {r:.2f}; same top parent {same:.2f} of {both.sum()} children (chance {1/3072:.4f})")
    # chain from n2283
    b, n, rows = 17, 2283, []
    while True:
        rows.append((b, n, int(T[b][:, n].sum()), int(TC[b][:, n].sum()) if b in TC else None))
        if b - 1 < 11: break
        has, best, _ = nested(T[b][:, [n]], T[b - 1])
        if not has[0]: break
        b, n = b - 1, int(best[0])
    o["chain"] = rows
    print("  chain from b17 n2283:", " <- ".join(f"b{b} n{n} ({ni} imgs, {nc} cls)" for b, n, ni, nc in rows))
    for b, n, _, _ in rows:
        top_cls = np.bincount(ya[T[b][:, n]], minlength=1000)
        print(f"    b{b} n{n} top classes of its images:", [names[k] for k in np.argsort(-top_cls)[:8]])
    out[key] = o
json.dump(out, open(os.path.join(RES, "vision-r9-inputs.json"), "w"), indent=1, default=str)
