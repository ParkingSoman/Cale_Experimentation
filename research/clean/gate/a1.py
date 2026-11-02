"""A1: placement check for sequential RCF (CHILDREN-LOG "RCF SCALE + CHANNELS", A1). Like g12, restricted to --layers.

place = fold    : g12 RCF (route parts added to layer L-1's write matrix and layer L's o_proj).
place = oracle  : same fitted route parts, but written AFTER MLP L (nothing in layer L reads them). Extra storage;
                  an upper bound only.
place = refit   : fold, then recapture layer L's inputs with the fold installed and refit ONLY the stale parts (kept
                  writes Dk, attention fold Ca) to a local drift-correcting target; the L-1 fold Cp stays fixed.
arm = cut       : top-K + LS (no routes).

  python3 research/clean/gate/a1.py --layers 11-20 --K 768 --place oracle
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from contract import orig_weights, ROOT, SwiGLULayer, SEQ  # noqa: E402
from tree_lib import load_model  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import silu  # noqa: E402
from g10 import Lin  # noqa: E402
from g12 import capture, dev_sets, mixed_cal  # noqa: E402
from sink import SinkGuard  # noqa: E402
from a0 import grams, cv_ridge_g, LAMS, FORBID  # noqa: E402

f32 = lambda a: mx.array(np.asarray(a, dtype=np.float32))


class Stash(nn.Module):
    """Transparent wrapper that stores silu(x G^T) * (x U^T) (route hiddens) in box[key]."""
    def __init__(self, inner, G, U, box, key):
        super().__init__(); self.inner, self.Gs, self.Us, self.box, self.key = inner, f32(G), f32(U), box, key
    def __call__(self, x):
        self.box[self.key] = nn.silu(x @ self.Gs.T) * (x @ self.Us.T); return self.inner(x)


class ZStash(nn.Module):
    def __init__(self, inner, box, key):
        super().__init__(); self.inner, self.box, self.key = inner, box, key
    def __call__(self, x):
        self.box[self.key] = x; return self.inner(x)


class AddAfter(nn.Module):
    """MLP L output + route writes taken from box (zero at block position 0, as the sink guard)."""
    def __init__(self, inner, parts, box):
        super().__init__(); self.inner, self.parts, self.box = inner, [(k, f32(C)) for k, C in parts], box
    def __call__(self, x):
        y = self.inner(x)
        for k, C in self.parts:
            y = y + (self.box[k] @ C.T) * (mx.arange(x.shape[1]) > 0)[None, :, None]
        return y


def parse_layers(s):
    if "-" in s:
        a, b = map(int, s.split("-")); return [l for l in range(a, b + 1) if l not in FORBID and l >= 1]
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="11-20"); ap.add_argument("--K", type=int, default=768)
    ap.add_argument("--place", default="fold"); ap.add_argument("--arm", default="rcf"); ap.add_argument("--ncal", type=int, default=128)
    ap.add_argument("--fold_on", default="", help="layers that get routes (default: all in --layers)"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); Ls = parse_layers(a.layers); assert not FORBID & set(Ls)
    fold_on = set(parse_layers(a.fold_on)) if a.fold_on else set(Ls)
    model, tok = load_model(); cal = mixed_cal(tok, a.ncal); dev, devc = dev_sets(tok)
    base_p, base_c = block_ce(model, dev), block_ce(model, devc)
    layers = model.model.layers; K = a.K; box = {}
    origw = {l: orig_weights(model, l) for l in range(len(layers))}; kept = {}; t0 = time.time()
    for L in Ls:
        lay = layers[L]; rcf = a.arm == "rcf" and L in fold_on
        mods = {"x": (lay, "mlp")}
        if rcf:
            mods["z"] = (lay.self_attn, "o_proj")
            if L - 1 >= 1:
                mods["xp"] = (layers[L - 1], "mlp")
        cap = capture(model, cal, mods)
        X = cap["x"]; G, U, D = origw[L]; Gf, Uf, Df = (w.astype(np.float32) for w in (G, U, D))
        H = (silu(X @ Gf.T) * (X @ Uf.T)).astype(np.float32); Y = (H @ Df.T).astype(np.float32)
        en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0); keep = np.sort(np.argsort(-en)[:K]); Hk = H[:, keep]; del H
        feats = [Hk]; pw = None
        if rcf and "xp" in cap:
            pw = kept[L - 1] if (L - 1) in kept else origw[L - 1]      # routes = the previous layer's CURRENT neurons
            feats.append((silu(cap["xp"] @ pw[0].T.astype(np.float32)) * (cap["xp"] @ pw[1].T.astype(np.float32))).astype(np.float32))
        if rcf:
            feats.append(cap["z"])
        A = np.concatenate(feats, 1); W, lam = cv_ridge_g(grams(A, Y), LAMS if rcf else (1e-4,)); del A
        Dk = W[:K].T; i = K; Cp = Ca = None
        if pw is not None:
            Cp = W[i:i + pw[0].shape[0]]; i += pw[0].shape[0]
        if rcf:
            Ca = W[i:]
        kept[L] = (G[keep], U[keep], Dk)
        lay.mlp = SinkGuard(SwiGLULayer(*kept[L]), SwiGLULayer(G, U, D))
        Wo = np.array(lay.self_attn.o_proj.weight, dtype=np.float64)
        if rcf and a.place == "oracle":
            parts = []
            if Cp is not None:
                layers[L - 1].mlp = Stash(layers[L - 1].mlp, pw[0], pw[1], box, f"p{L}"); parts.append((f"p{L}", Cp.T))
            lay.self_attn.o_proj = ZStash(lay.self_attn.o_proj, box, f"z{L}"); parts.append((f"z{L}", Ca.T))
            lay.mlp = AddAfter(lay.mlp, parts, box)
        elif rcf:
            if Cp is not None:
                Gp, Up, Dp = pw
                newp = SinkGuard(SwiGLULayer(Gp, Up, Dp + Cp.T), SwiGLULayer(*origw[L - 1]))
                layers[L - 1].mlp = newp
                if (L - 1) in kept:
                    kept[L - 1] = (Gp, Up, Dp + Cp.T)
            lay.self_attn.o_proj = SinkGuard(Lin(Wo + Ca.T), Lin(Wo))
            if a.place == "refit":
                # local drift-correcting target: (Dk hk' + Ca z') = Y_pre - Cp hp - Wo (z' - z)
                cap2 = capture(model, cal, {"x": (lay, "mlp"), "z": (lay.self_attn, "o_proj")})
                Hk2 = (silu(cap2["x"] @ Gf[keep].T) * (cap2["x"] @ Uf[keep].T)).astype(np.float32)
                T = Y - (cap2["z"] - cap["z"]) @ Wo.T.astype(np.float32)
                if Cp is not None:
                    Hp = feats[1]; T = T - Hp @ Cp.astype(np.float32)
                A2 = np.concatenate([Hk2, cap2["z"]], 1); W2, lam2 = cv_ridge_g(grams(A2, T.astype(np.float32)), LAMS); del A2
                kept[L] = (G[keep], U[keep], W2[:K].T)
                lay.mlp = SinkGuard(SwiGLULayer(*kept[L]), SwiGLULayer(G, U, D))
                lay.self_attn.o_proj = SinkGuard(Lin(Wo + W2[K:].T), Lin(Wo)); del cap2, Hk2, T
        del feats, cap, X, Y
        print(f"  L{L} done lam {lam} [{time.time()-t0:.0f}s]", flush=True)
    ce_p, ce_c = block_ce(model, dev), block_ce(model, devc)
    row = {"K": K, "layers": a.layers, "arm": a.arm, "place": a.place, "fold_on": a.fold_on, "ncal": a.ncal,
           "dce_prose": float((ce_p - base_p).mean()), "dce_code": float((ce_c - base_c).mean()),
           "ce_prose_blocks": ce_p.tolist(), "ce_code_blocks": ce_c.tolist()}
    path = os.path.join(ROOT, "results", "clean-a1.json"); out = json.load(open(path)) if os.path.exists(path) else {}
    key = f"{a.arm}_{a.place}_K{K}_L{a.layers}{'_on' + a.fold_on if a.fold_on else ''}_n{a.ncal}{a.tag}"
    out[key] = row; json.dump(out, open(path, "w"), indent=1)
    print(key, "dCE prose", round(row["dce_prose"], 4), "code", round(row["dce_code"], 4), f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
