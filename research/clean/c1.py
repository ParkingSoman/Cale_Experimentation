"""C1: per-word dial. Keep only the strongest MLP neurons for each token. No training.

Run: python3 research/clean/c1.py [--blocks N]   (N held-out blocks, default 100)
Writes results/clean-c1.json. Imports tree_lib read-only.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "research", "tree"))
from tree_lib import load_model, token_blocks  # noqa: E402

FRACS = [1.0, 0.5, 0.25, 0.125, 0.0625]


class TopKMLP:
    """Replaces an MLP. Keeps the top `frac` neurons per token and zeroes the rest.

    mode "hidden": rank by |silu(gate) * up|. Needs gate and up for every neuron, saves only down reads.
    mode "gate":   rank by |silu(gate)|. Needs only gate for every neuron, saves up and down reads.
    mode "dneg":   zero every activation with gate < 0 and up < 0 (the double-negative regime).
    """

    def __init__(self, inner, frac=1.0, mode="hidden"):
        self.inner, self.frac, self.mode = inner, frac, mode
        self.stats = None  # set to a list to collect (share of top-25% that is double-negative)

    def __call__(self, x):
        m = self.inner
        g, u = m.gate_proj(x), m.up_proj(x)
        sg = g * mx.sigmoid(g)
        h = sg * u
        n = h.shape[-1]
        if self.stats is not None:
            score = mx.abs(h)
            k = max(1, int(0.25 * n))
            thr = mx.sort(score, axis=-1)[..., n - k][..., None]
            top = score >= thr
            dn = (g < 0) & (u < 0)
            self.stats.append(((top & dn).sum() / top.sum()).item())
        if self.mode == "dneg":
            h = mx.where((g < 0) & (u < 0), 0.0, h)
        elif self.frac < 1.0:
            k = max(1, int(round(self.frac * n)))
            score = mx.abs(sg) if self.mode == "gate" else mx.abs(h)
            thr = mx.sort(score, axis=-1)[..., n - k][..., None]
            h = mx.where(score >= thr, h, 0.0)
        return m.down_proj(h)


def set_mlps(model, originals, specs):
    """specs: list of (frac, mode) per layer, or None to restore the original MLPs."""
    for i, layer in enumerate(model.model.layers):
        layer.mlp = originals[i] if specs is None else TopKMLP(originals[i], *specs[i])


def per_block(model, teacher_logp, blocks, batch=4):
    """Per-block mean CE and KL(teacher || model). teacher_logp: list per batch."""
    ce, kl = [], []
    for bi, i in enumerate(range(0, blocks.shape[0], batch)):
        b = blocks[i:i + batch]
        x, y = b[:, :-1], b[:, 1:]
        lp = model(x)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        c = -mx.take_along_axis(lp, y[..., None], axis=-1).squeeze(-1).mean(-1)
        tl = teacher_logp[bi]
        k = (mx.exp(tl) * (tl - lp)).sum(-1).mean(-1)
        mx.eval(c, k)
        ce += np.array(c).tolist()
        kl += np.array(k).tolist()
    return np.array(ce), np.array(kl)


def teacher_cache(model, blocks, batch=4):
    out = []
    for i in range(0, blocks.shape[0], batch):
        lp = model(blocks[i:i + batch, :-1])
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        mx.eval(lp)
        out.append(lp)
    return out


def summarize(ce, kl, rng):
    boot = [ce[rng.integers(0, len(ce), len(ce))].mean() for _ in range(200)]
    return {"ce": float(ce.mean()), "kl": float(kl.mean()), "ce_noise": float(np.std(boot))}


def bytes_read(frac, mode):
    """Share of the MLP weight bytes read per token (gate, up, down are equal in size)."""
    if mode == "gate":
        return (1 + 2 * frac) / 3
    return (2 + frac) / 3


def budgets(sens, mean_frac, lo=1 / 64):
    """Per-layer fractions proportional to sqrt(sensitivity), mean fixed, each in [lo, 1]."""
    w = np.sqrt(np.maximum(sens, 1e-9))
    f = w / w.mean() * mean_frac
    for _ in range(50):
        f = np.clip(f, lo, 1.0)
        f *= mean_frac / f.mean()
    return np.clip(f, lo, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=100)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)

    model, tok = load_model()
    cal = token_blocks(tok, "train", 32)
    held = token_blocks(tok, "test", 100)[: args.blocks]
    originals = [l.mlp for l in model.model.layers]
    n_layers = len(originals)

    set_mlps(model, originals, None)
    t_held = teacher_cache(model, held)
    cal8 = cal[:8]
    t_cal8 = teacher_cache(model, cal8)
    base_ce, base_kl = per_block(model, t_held, held)
    print(f"baseline held-out CE {base_ce.mean():.4f}  (expect 3.0180 with 100 blocks)", flush=True)
    res = {"baseline": summarize(base_ce, base_kl, rng), "held_blocks": int(held.shape[0]), "runs": []}

    def run(label, specs, mean_frac, mode):
        set_mlps(model, originals, specs)
        ce, kl = per_block(model, t_held, held)
        set_mlps(model, originals, None)
        r = {"label": label, "mode": mode, "mean_frac": mean_frac, "bytes_read": bytes_read(mean_frac, mode),
             **summarize(ce, kl, rng)}
        res["runs"].append(r)
        print(f"{label:28s} CE {r['ce']:.4f} (+/-{r['ce_noise']:.4f})  KL {r['kl']:.4f}  bytes {r['bytes_read']:.2f}"
              f"  [{time.time() - t0:.0f}s]", flush=True)

    # 1. Uniform dial, two ranking rules.
    for mode in ("hidden", "gate"):
        for f in FRACS:
            run(f"{mode} uniform {f}", [(f, mode)] * n_layers, f, mode)

    # 2. Per-layer sensitivity on calibration text (8 blocks), then a budget with the same mean.
    sens = []
    for i in range(n_layers):
        specs = [(1.0, "hidden")] * n_layers
        specs[i] = (0.25, "hidden")
        set_mlps(model, originals, specs)
        _, kl = per_block(model, t_cal8, cal8)
        sens.append(float(kl.mean()))
    set_mlps(model, originals, None)
    res["layer_sensitivity_kl_at_0.25"] = sens
    print("layer sensitivity (KL, calibration, one layer at 25%):", [round(s, 4) for s in sens], flush=True)
    for mean_f in (0.25, 0.125):
        fr = budgets(np.array(sens), mean_f)
        res.setdefault("budget_fracs", {})[str(mean_f)] = fr.tolist()
        run(f"hidden budget mean {mean_f}", [(float(x), "hidden") for x in fr], mean_f, "hidden")

    # 3. Essay fact 3: the double-negative regime.
    set_mlps(model, originals, [(0.25, "hidden")] * n_layers)
    for l in model.model.layers:
        l.mlp.stats = []
    per_block(model, t_cal8, cal8)
    share = [float(np.mean(l.mlp.stats)) for l in model.model.layers]
    set_mlps(model, originals, None)
    res["double_negative_share_of_top25"] = share
    print("double-negative share of top-25% activations, by layer:", [round(s, 3) for s in share], flush=True)
    run("zero all double-negative", [(1.0, "dneg")] * n_layers, 1.0, "dneg")

    # 4. Static cut from the other session (Z2: cut_act + least squares), for comparison.
    z2 = json.load(open(os.path.join(ROOT, "results", "tree-z2.json")))
    res["static_cut_z2"] = {
        str(r["size"] / 1536): {"ce": r["ce"], "kl": r["kl"]}
        for r in z2["results"] if r["method"] == "cut_act" and r["ls"]
    }
    print("static cut (Z2, cut_act+LS):", res["static_cut_z2"], flush=True)

    out = os.path.join(ROOT, "results", "clean-c1.json")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out, f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
