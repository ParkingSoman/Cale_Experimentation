"""G12: whole-model MLP coarsening with route-capture folds. Sequential (first to last), each layer fitted on inputs
from the already-coarsened model; target = the ORIGINAL MLP applied to those inputs. Coarsened layers: 1-29 except the
keeper layers 6, 9, 21, 24 (layer 0 stays intact). Mixed calibration. Sink guard on every changed module.

  python3 research/clean/gate/g12.py --frac 0.5 --arm rcf      (arm: cut | af | rcf)
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from contract import orig_weights, ROOT, SwiGLULayer, SEQ  # noqa: E402
from tree_lib import load_model, token_blocks  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import silu, boot  # noqa: E402
from g7 import cv_ridge  # noqa: E402
from g10 import Lin  # noqa: E402
from sink import SinkGuard  # noqa: E402

SKIP = {0, 6, 9, 21, 24}


class Grab(nn.Module):
    def __init__(self, inner):
        super().__init__(); self.inner, self.x = inner, []
    def __call__(self, x):
        self.x.append(np.array(x).reshape(-1, x.shape[-1]).astype(np.float32)); return self.inner(x)


def capture(model, blocks, mods, batch=4):
    """mods: dict name -> (parent, attr). Returns name -> rows (block position 0 dropped)."""
    gs = {}
    for k, (par, at) in mods.items():
        gs[k] = Grab(getattr(par, at)); setattr(par, at, gs[k])
    for i in range(0, blocks.shape[0], batch):
        mx.eval(model(blocks[i:i + batch, :-1]))
    out = {}
    for k, (par, at) in mods.items():
        setattr(par, at, gs[k].inner); X = np.concatenate(gs[k].x, 0); out[k] = X[np.arange(len(X)) % SEQ != 0]
    return out


def dev_sets(tok):
    dev = token_blocks(tok, "test", 40)
    txt = open(os.path.join(ROOT, "research", "clean", "data", "code_train.txt")).read()
    ids = np.array(tok.encode(txt[:400000]), dtype=np.int32); nb = min(40, len(ids) // 513)
    return dev, mx.array(ids[:nb * 513].reshape(nb, 513))


def mixed_cal(tok, n=32):
    cal = token_blocks(tok, "train", n)
    txt = open(os.path.join(ROOT, "research", "clean", "data", "code_train.txt")).read()[400000:]
    ids = np.array(tok.encode(txt[:800000]), dtype=np.int32)[: n * 513]
    return mx.concatenate([cal, mx.array(ids.reshape(n, 513))], 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frac", type=float, default=0.5); ap.add_argument("--arm", default="rcf")
    ap.add_argument("--ncal", type=int, default=32); ap.add_argument("--tag", default=""); ap.add_argument("--dc", action="store_true"); ap.add_argument("--layers", default=""); ap.add_argument("--K", type=int, default=0); ap.add_argument("--fold_from", type=int, default=0); ap.add_argument("--fold_to", type=int, default=99)
    a = ap.parse_args()
    model, tok = load_model(); cal = mixed_cal(tok, a.ncal); dev, devc = dev_sets(tok)
    base_p, base_c = block_ce(model, dev), block_ce(model, devc)
    layers = model.model.layers; n = layers[0].mlp.gate_proj.weight.shape[0]; K = a.K or int(round(a.frac * n))
    only = set(int(x) for x in a.layers.split(",")) if a.layers else None
    kept = {}          # l -> (G, U, D) of the coarse layer currently installed
    origw = {l: orig_weights(model, l) for l in range(len(layers))}
    t0 = time.time()
    if a.dc:   # original residual after each layer (block position 0 dropped)
        outs = {}
        class OutGrab(nn.Module):
            def __init__(self, inner, l):
                super().__init__(); self.inner, self.l = inner, l; self.use_sliding = getattr(inner, "use_sliding", False)
            def __call__(self, *args, **kw):
                y = self.inner(*args, **kw); outs.setdefault(self.l, []).append(np.array(y).reshape(-1, y.shape[-1]).astype(np.float32)); return y
        saved = list(layers)
        for l in range(len(layers)):
            model.model.layers[l] = OutGrab(saved[l], l)
        for i in range(0, cal.shape[0], 4):
            mx.eval(model(cal[i:i + 4, :-1]))
        for l in range(len(layers)):
            model.model.layers[l] = saved[l]
        layers = model.model.layers
        h_orig = {l: (lambda X: X[np.arange(len(X)) % SEQ != 0])(np.concatenate(v, 0)) for l, v in outs.items()}; del outs
        print(f"original residuals ready [{time.time()-t0:.0f}s]", flush=True)
    for L in range(len(layers)):
        if L in SKIP or (only is not None and L not in only):
            continue
        lay = layers[L]; mods = {"x": (lay, "mlp")}
        arm = a.arm if a.fold_from <= L <= a.fold_to else "cut"
        if arm in ("af", "rcf"):
            mods["z"] = (lay.self_attn, "o_proj")
        prev = L - 1 if (arm in ("rcf", "plf") and (L - 1) in kept) else None
        prev_full = arm in ("rcf", "plf") and prev is None and (L - 1) >= 1 and (L - 1) not in SKIP
        if prev is not None or prev_full:
            mods["xp"] = (layers[L - 1], "mlp")
        if a.dc:
            mods["hmid"] = (lay, "post_attention_layernorm")
        cap = capture(model, cal, mods)
        X = cap["x"]; G, U, D = origw[L]
        H = silu(X @ G.T) * (X @ U.T); Y = (H @ D.T).astype(np.float32)
        if a.dc:
            Y = (h_orig[L] - cap["hmid"]).astype(np.float32)
        en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0); keep = np.sort(np.argsort(-en)[:K])
        feats = [H[:, keep].astype(np.float32)]
        if prev is not None:
            Gp, Up, Dp = kept[prev]; feats.append((silu(cap["xp"] @ Gp.T) * (cap["xp"] @ Up.T)).astype(np.float32))
        elif prev_full:   # previous layer is NOT coarsened: all its neurons are routes
            Gp, Up, Dp = origw[L - 1]; feats.append((silu(cap["xp"] @ Gp.T) * (cap["xp"] @ Up.T)).astype(np.float32))
        if "z" in cap:
            feats.append(cap["z"])
        W, lam = cv_ridge(np.concatenate(feats, 1), Y, (1e-3, 1e-2, 1e-1) if len(feats) > 1 else (1e-4,))
        Dk = W[:K].T; i = K
        kept[L] = (G[keep], U[keep], Dk)
        lay.mlp = SinkGuard(SwiGLULayer(*kept[L]), SwiGLULayer(G, U, D))
        if prev is not None:
            Cp = W[i:i + K]; i += K
            Gp, Up, Dp = kept[prev]; kept[prev] = (Gp, Up, Dp + Cp.T)
            layers[prev].mlp = SinkGuard(SwiGLULayer(*kept[prev]), SwiGLULayer(*origw[prev]))
        elif prev_full:
            Gp, Up, Dp = origw[L - 1]; Cp = W[i:i + Gp.shape[0]]; i += Gp.shape[0]
            layers[L - 1].mlp = SinkGuard(SwiGLULayer(Gp, Up, Dp + Cp.T), SwiGLULayer(Gp, Up, Dp))
        if "z" in cap:
            Ca = W[i:]; Wo = np.array(lay.self_attn.o_proj.weight if not isinstance(lay.self_attn.o_proj, SinkGuard) else lay.self_attn.o_proj.orig.W, dtype=np.float64)
            lay.self_attn.o_proj = SinkGuard(Lin(Wo + Ca.T), Lin(Wo))
        print(f"  L{L} done lam {lam} [{time.time()-t0:.0f}s]", flush=True)
    ce_p, ce_c = block_ce(model, dev), block_ce(model, devc)
    row = {"frac": a.frac, "arm": a.arm, "K": K, "dce_prose": float((ce_p - base_p).mean()), "dce_code": float((ce_c - base_c).mean()),
           "ce_prose_blocks": ce_p.tolist(), "ce_code_blocks": ce_c.tolist(), "base_prose": float(base_p.mean()), "base_code": float(base_c.mean())}
    path = os.path.join(ROOT, "results", "clean-g12.json"); out = json.load(open(path)) if os.path.exists(path) else {}
    out[f"{a.arm}_f{a.frac}_K{K}_L{a.layers or 'all'}{'_dc' if a.dc else ''}{'_ff' + str(a.fold_from) if a.fold_from else ''}{'_ft' + str(a.fold_to) if a.fold_to < 99 else ''}{a.tag}"] = row; json.dump(out, open(path, "w"), indent=1)
    print(a.arm, a.frac, "dc" if a.dc else "", "dCE prose", round(row["dce_prose"], 4), "code", round(row["dce_code"], 4), flush=True)


if __name__ == "__main__":
    main()
