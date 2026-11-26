"""Controls for fit.py (advisor review, 2026-11-13). Is the absorbed share the PARENT's, or would anything do?

1. bias only: fit a with b = 0 (same convex KL solve).
2. other channels: fit a + b * g_q for the two other parents and 20 random channels q. Rank the parent among them.
3. cross table: drop cost of every recipe (6 parent recipes + 2 random removals) on every goal.
4. paired bootstrap on own20 rows: parent column against bias only, closed form against KL optimum.
All scores are mean KL on test photos (odd positions inside each class).
"""
import json
import os

import numpy as np
import torch

from fit import kl_rows, lbfgs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
torch.set_default_dtype(torch.float64)


def solve(yN, yD, feat):
    """KL optimum of yD + a (+ b * feat). Returns (a, b)."""
    A = torch.zeros(yN.shape[1], requires_grad=True)
    if feat is None:
        with torch.enable_grad():
            lbfgs([A], lambda: kl_rows(yN, yD + A).mean(), iters=200)
        return A.detach(), None
    B = torch.zeros(yN.shape[1], requires_grad=True)
    with torch.enable_grad():
        lbfgs([A, B], lambda: kl_rows(yN, yD + A + B * feat[:, None]).mean(), iters=200)
    return A.detach(), B.detach()


def boot(diff, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = diff.numpy()
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def main():
    d = np.load(os.path.join(OUT, "absorb_cache.npz"))
    r = np.load(os.path.join(OUT, "rand_cache.npz"))
    prev = {(x["parent"], x["recipe"], x["goal"]): x for x in json.load(open(os.path.join(OUT, "absorb_scores.json")))}
    W, bias = torch.tensor(d["fcW"]).double(), torch.tensor(d["fcb"]).double()
    lab, fit = d["lab"], d["fit"]
    yN = torch.tensor(d["gN"]).double() @ W.T + bias
    parents = [245, 418, 237]
    gD = {f"{p}_{rc}": torch.tensor(d[f"gD_{p}_{rc}"]).double() for p in parents for rc in ("map", "edge")}
    gD.update({k: torch.tensor(r[f"gD_{k}"]).double() for k in ("rand3", "rand10")})
    goals = {"all": np.ones(len(lab), bool)}
    for pi, p in enumerate(parents):
        par = d["par"][:, pi]
        cm = np.array([par[(lab == c) & fit].mean() for c in range(1000)])
        goals[f"own20_{p}"] = np.isin(lab, np.argsort(-cm)[:20])
    out = dict(cross={}, rows=[])

    # 3. cross table of drop cost
    for rk, g in gD.items():
        yD = g @ W.T + bias
        out["cross"][rk] = {gn: float(kl_rows(yN[gm & ~fit], yD[gm & ~fit]).mean()) for gn, gm in goals.items()}
        print("cross", rk, {k: round(v, 5) for k, v in out["cross"][rk].items()}, flush=True)

    rng = np.random.default_rng(1)
    rand_q = sorted(rng.choice([q for q in range(512) if q not in parents], 20, replace=False).tolist())
    rows = [(p, rc, gn) for p in parents for rc in ("map", "edge") for gn in ("all", f"own20_{p}")]
    rows += [(245, "rand3", "own20_245"), (245, "rand10", "own20_245"), (245, "rand3", "all"), (245, "rand10", "all")]
    for p, rc, gn in rows:
        key = rc if rc.startswith("rand") else f"{p}_{rc}"
        g = gD[key]
        yD = g @ W.T + bias
        gm = goals[gn]
        F, T = np.where(gm & fit)[0], np.where(gm & ~fit)[0]
        drop_t = kl_rows(yN[T], yD[T])
        a0, _ = solve(yN[F], yD[F], None)
        bias_t = kl_rows(yN[T], yD[T] + a0)
        res = dict(parent=p, recipe=rc, goal=gn, drop=float(drop_t.mean()), bias_only=float(bias_t.mean()), chan={})
        per_photo = {}
        for q in [p] + [o for o in parents if o != p] + rand_q:
            a, b = solve(yN[F], yD[F], g[F, q])
            t = kl_rows(yN[T], yD[T] + a + b * g[T, q][:, None])
            res["chan"][q] = float(t.mean())
            per_photo[q] = t
        others = [res["chan"][q] for q in rand_q]
        res["parent_rank"] = 1 + sum(v < res["chan"][p] for q, v in res["chan"].items() if q != p)
        res["rand_median"], res["rand_best"] = float(np.median(others)), float(min(others))
        if gn.startswith("own20"):
            res["boot_parent_minus_bias"] = boot(per_photo[p] - bias_t)
            res["boot_parent_minus_randmedian"] = boot(per_photo[p] - per_photo[rand_q[int(np.argsort(others)[len(others) // 2])]])
            pv = prev.get((p, rc, "own20"))
            if pv is not None:
                # closed form against KL optimum needs the closed per-photo losses: recompute the closed fit here
                yF, yDF, hp = yN[F], yD[F], g[F, p]
                gap = yF - yDF
                gap = gap - gap.mean(1, keepdim=True)
                hm = hp.mean()
                bb = ((hp - hm)[:, None] * (gap - gap.mean(0))).mean(0) / ((hp - hm) ** 2).mean()
                aa = gap.mean(0) - bb * hm
                closed_t = kl_rows(yN[T], yD[T] + aa + bb * g[T, p][:, None])
                res["boot_closed_minus_klopt"] = boot(closed_t - per_photo[p])
        out["rows"].append(res)
        show = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items() if k != "chan"}
        show["parent_col"] = round(res["chan"][p], 5)
        print(json.dumps(show), flush=True)
    out["rand_q"] = rand_q
    json.dump(out, open(os.path.join(OUT, "controls.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
