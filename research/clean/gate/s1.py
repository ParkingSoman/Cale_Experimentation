"""S1: can one SHARED set of layer L-1 gate/up changes serve several widths of layer L? (CHILDREN-LOG S1)
Same setup as v2 (single layer L, fold placement, 3-layer window from the input of layer L-1). Two widths trained
together, each step = loss at width 1 + loss at width 2. Shared: layer L-1 MLP gate G and up U. Per width: kept writes
at L, the L-1 write matrix (it holds the width's fold), the L o_proj fold. Compare with v3's per-width Dprev.

  python3 research/clean/gate/s1.py --L 16 --Ks 48,768 --secs 300
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx_lm.models.base import create_attention_mask
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from a0 import grams, cv_ridge_g, LAMS, ROOT  # noqa: E402
from q1 import setup  # noqa: E402
from a2 import grab_blocks  # noqa: E402
from contract import SwiGLULayer  # noqa: E402
from g10 import Lin  # noqa: E402
from sink import SinkGuard  # noqa: E402
from tree_lib import load_model  # noqa: E402
from r1 import block_ce  # noqa: E402
from g12 import dev_sets, mixed_cal  # noqa: E402

f32 = lambda a: mx.array(np.asarray(a, dtype=np.float32))


class SharedPrev(nn.Module):
    """Layer L-1 MLP with shared G, U and one write matrix per width; .w picks the width."""
    def __init__(self, Gp, Up, Ds, orig):
        super().__init__(); self.G, self.U = f32(Gp), f32(Up); self.Ds = [f32(D) for D in Ds]; self.orig = orig; self.w = 0
    def __call__(self, x):
        y = (nn.silu(x @ self.G.T) * (x @ self.U.T)) @ self.Ds[self.w].T
        return mx.concatenate([self.orig(x[:, :1]), y[:, 1:]], axis=1)


class Holder(nn.Module):
    def __init__(self, prev, mlps, ops):
        super().__init__(); self.prev, self.mlps, self.ops = prev, mlps, ops


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--L", type=int, default=16); ap.add_argument("--Ks", default="48,768")
    ap.add_argument("--secs", type=float, default=300); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--window", type=int, default=3); ap.add_argument("--balance", action="store_true")
    a = ap.parse_args(); L = a.L; Ks = [int(k) for k in a.Ks.split(",")]
    model, tok = load_model(); cal = mixed_cal(tok, 128); dev, devc = dev_sets(tok)
    fitb = mx.concatenate([cal[:64], cal[128:192]], 0); texts = (("prose", dev), ("code", devc))
    base = {t: block_ce(model, b) for t, b in texts}
    m = model.model.layers; up = min(L + a.window, len(m) - 1)
    hin, _ = grab_blocks(model, fitb, L - 1, None); mask = create_attention_mask(mx.array(hin[:1]), None)

    def thru(h):
        for l in range(L - 1, up + 1):
            h = m[l](h, mask)
        return h
    tg = np.concatenate([np.array(thru(mx.array(hin[j:j + 4]))) for j in range(0, hin.shape[0], 4)], 0)
    s = setup(model, L, cal); G, U, D = s["W"]; Gp, Up, Dp = s["Wp"]; P = s["Hp"].shape[1]
    R = np.concatenate([s["Hp"], s["Z"]], 1); order = np.argsort(-s["en"])
    lay, prev = m[L], m[L - 1]; saved = (lay.mlp, lay.self_attn.o_proj, prev.mlp)
    Wo = np.array(saved[1].weight, dtype=np.float64); origL, origP = SwiGLULayer(G, U, D), SwiGLULayer(Gp, Up, Dp)
    keeps, Wcs, mlps, ops, Ds = [], [], [], [], []
    for K in Ks:
        keep = np.sort(order[:K]); Hk = s["H"][:, keep]
        Wc, _ = cv_ridge_g(grams(Hk, s["Y"]), (1e-4,)); W, _ = cv_ridge_g(grams(np.concatenate([Hk, R], 1), s["Y"]), LAMS)
        keeps.append(keep); Wcs.append(Wc)
        mlps.append(SinkGuard(SwiGLULayer(G[keep], U[keep], W[:K].T), origL)); ops.append(SinkGuard(Lin(Wo + W[K + P:].T), Lin(Wo)))
        Ds.append(Dp + W[K:K + P].T)
    sp = SharedPrev(Gp, Up, Ds, origP); hold = Holder(sp, mlps, ops)

    def use(i):
        sp.w = i; prev.mlp, lay.mlp, lay.self_attn.o_proj = sp, mlps[i], ops[i]

    def score_all():
        res = {}
        for i, K in enumerate(Ks):
            use(i); res[K] = {t: float((block_ce(model, b) - base[t]).mean()) for t, b in texts}
        return res

    cut = {}
    for i, K in enumerate(Ks):
        lay.mlp = SinkGuard(SwiGLULayer(G[keeps[i]], U[keeps[i]], Wcs[i].T), origL); prev.mlp, lay.self_attn.o_proj = saved[2], saved[1]
        cut[K] = {t: float((block_ce(model, b) - base[t]).mean()) for t, b in texts}
    start = score_all()
    model.freeze(); hold.unfreeze(); hold.prev.orig.freeze()
    for g in hold.mlps:
        g.orig.freeze(); g.new.freeze(keys=["G", "U"])
    for o in hold.ops:
        o.orig.freeze()

    def one(i, h, t):
        use(i); r = (thru(h) - t)[:, 1:]; c = t[:, 1:] - t[:, 1:].mean(-1, keepdims=True); return (r * r).sum() / (c * c).sum()
    scale = [1.0] * len(Ks)
    if a.balance:   # each width's loss divided by its start value
        r0 = np.random.default_rng(99); bs = [r0.integers(0, hin.shape[0], 4) for _ in range(8)]
        scale = [float(np.mean([float(one(i, mx.array(hin[b]), mx.array(tg[b]))) for b in bs])) for i in range(len(Ks))]
        print("start losses", scale, flush=True)

    def loss_fn(hd, h, t):
        tot = 0.0
        for i in range(len(Ks)):
            tot = tot + one(i, h, t) / scale[i]
        return tot
    vg = nn.value_and_grad(hold, loss_fn); opt = optim.Adam(learning_rate=a.lr); rng = np.random.default_rng(L); t0 = time.time(); n = 0
    while time.time() - t0 < a.secs:
        idx = rng.integers(0, hin.shape[0], 4)
        _, g = vg(hold, mx.array(hin[idx]), mx.array(tg[idx])); opt.update(hold, g); mx.eval(hold.parameters(), opt.state); n += 1
    model.unfreeze()
    end = score_all(); lay.mlp, lay.self_attn.o_proj, prev.mlp = saved
    row = {"steps": n, "secs": a.secs}
    for K in Ks:
        row[K] = {t: {"start_share": 1 - start[K][t] / cut[K][t], "shared_share": 1 - end[K][t] / cut[K][t]} for t, _ in texts}
    path = os.path.join(ROOT, "results", "clean-s1.json"); out = json.load(open(path)) if os.path.exists(path) else {}
    out[f"L{L}_K{a.Ks}{'_bal' if a.balance else ''}"] = row; json.dump(out, open(path, "w"), indent=1)
    for K in Ks:
        print(f"L{L} K{K} shared L-1 gates: share prose {row[K]['prose']['shared_share']:.0%} code {row[K]['code']['shared_share']:.0%}"
              f" (start {row[K]['prose']['start_share']:.0%}/{row[K]['code']['start_share']:.0%}), {n} joint steps", flush=True)


if __name__ == "__main__":
    main()
