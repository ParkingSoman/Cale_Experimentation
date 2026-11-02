"""A2: sequential whole-model coarsening with a 3-layer downstream window (CHILDREN-LOG "RCF SCALE + CHANNELS", A2).

Per layer L (first to last): closed-form start as in a1 (fold placement), then a short Adam fit of the free parts so the
residual after layer L+w matches the teacher. Free parts: arm rcf = kept writes Dk, L-1 fold Cp, attention fold Ca;
arm cut = kept writes Dk only (fairness control). Teacher = ORIGINAL layers L..L+w on the same (drifted) input of
layer L. Layers after L are still original when L is fitted (sequential).

  python3 research/clean/gate/a2.py --frac 0.5 --arm rcf --fold_on 10,11,12 --minutes 1.5
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx_lm.models.base import create_attention_mask
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from contract import orig_weights, ROOT, SwiGLULayer  # noqa: E402
from tree_lib import load_model  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import silu  # noqa: E402
from g10 import Lin  # noqa: E402
from g12 import capture, dev_sets, mixed_cal  # noqa: E402
from sink import SinkGuard  # noqa: E402
from a0 import grams, cv_ridge_g, LAMS, FORBID  # noqa: E402
from a1 import parse_layers  # noqa: E402

f32 = lambda a: mx.array(np.asarray(a, dtype=np.float32))


class InGrab(nn.Module):
    """Wraps a decoder layer; stores its input residual (full blocks)."""
    def __init__(self, inner):
        super().__init__(); self.inner, self.x = inner, []; self.use_sliding = getattr(inner, "use_sliding", False)
    def __call__(self, x, mask=None, cache=None):
        self.x.append(np.array(x)); return self.inner(x, mask, cache=cache)


class HidGrab(nn.Module):
    """Wraps an MLP; stores silu(x G^T) * (x U^T) for the given neurons (full blocks)."""
    def __init__(self, inner, G, U):
        super().__init__(); self.inner, self.Gs, self.Us, self.h = inner, f32(G), f32(U), []
    def __call__(self, x):
        self.h.append(np.array(nn.silu(x @ self.Gs.T) * (x @ self.Us.T))); return self.inner(x)


class FoldO(nn.Module):
    """o_proj = W0 (frozen) + C (trainable), C masked at block position 0 (sink guard)."""
    def __init__(self, W0, C):
        super().__init__(); self.W0, self.C = f32(W0), f32(C); self.freeze(keys=["W0"])
    def __call__(self, z):
        return z @ self.W0.T + (z @ self.C.T) * (mx.arange(z.shape[1]) > 0)[None, :, None]


class KeptMLP(nn.Module):
    """Kept neurons (G, U frozen, D trainable); original MLP output at block position 0."""
    def __init__(self, G, U, D, orig):
        super().__init__(); self.G, self.U, self.D = f32(G), f32(U), f32(D); self.orig = orig; self.freeze(keys=["G", "U"])
    def __call__(self, x):
        y = (nn.silu(x @ self.G.T) * (x @ self.U.T)) @ self.D.T
        y0 = mx.stop_gradient(self.orig(x[:, :1]))
        return mx.concatenate([y0, y[:, 1:]], axis=1)


class Student(nn.Module):
    def __init__(self, mlp, o=None, Cp=None):
        super().__init__(); self.mlp = mlp
        if o is not None:
            self.o = o
        if Cp is not None:
            self.Cp = f32(Cp)


def grab_blocks(model, blocks, L, pw, batch=4):
    """Input residual of layer L, and (optional) hiddens of layer L-1's route neurons, as [B, SEQ, *] arrays."""
    m = model.model; g = InGrab(m.layers[L]); m.layers[L] = g; hg = None
    if pw is not None:
        hg = HidGrab(m.layers[L - 1].mlp, pw[0], pw[1]); m.layers[L - 1].mlp = hg
    for i in range(0, blocks.shape[0], batch):
        mx.eval(model(blocks[i:i + batch, :-1]))
    m.layers[L] = g.inner
    if hg is not None:
        m.layers[L - 1].mlp = hg.inner
    return np.concatenate(g.x, 0), (np.concatenate(hg.h, 0) if hg is not None else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frac", type=float, default=0.5); ap.add_argument("--arm", default="rcf")
    ap.add_argument("--layers", default="1-29"); ap.add_argument("--fold_on", default=""); ap.add_argument("--ncal", type=int, default=128)
    ap.add_argument("--nfit", type=int, default=64, help="blocks per text for the window fit"); ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=1.5); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--tag", default="")
    a = ap.parse_args(); Ls = parse_layers(a.layers); fold_on = set(parse_layers(a.fold_on)) if a.fold_on else set(Ls)
    model, tok = load_model(); cal = mixed_cal(tok, a.ncal); dev, devc = dev_sets(tok)
    fitb = mx.concatenate([cal[:a.nfit], cal[a.ncal:a.ncal + a.nfit]], 0)       # prose + code blocks for the window fit
    base_p, base_c = block_ce(model, dev), block_ce(model, devc)
    m = model.model; layers = m.layers; n = layers[0].mlp.gate_proj.weight.shape[0]; K = int(round(a.frac * n))
    origw = {l: orig_weights(model, l) for l in range(len(layers))}; kept = {}; t0 = time.time(); log = []
    for L in Ls:
        lay = layers[L]; rcf = a.arm == "rcf" and L in fold_on
        pw = (kept[L - 1] if (L - 1) in kept else origw[L - 1]) if (rcf and L - 1 >= 1) else None
        # 1) closed-form start (as a1, fold placement)
        mods = {"x": (lay, "mlp")}
        if rcf:
            mods["z"] = (lay.self_attn, "o_proj")
            if pw is not None:
                mods["xp"] = (layers[L - 1], "mlp")
        cap = capture(model, cal, mods)
        X = cap["x"]; G, U, D = origw[L]; Gf, Uf, Df = (w.astype(np.float32) for w in (G, U, D))
        H = (silu(X @ Gf.T) * (X @ Uf.T)).astype(np.float32); Y = (H @ Df.T).astype(np.float32)
        en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0); keep = np.sort(np.argsort(-en)[:K])
        feats = [H[:, keep]]; del H
        if pw is not None:
            feats.append((silu(cap["xp"] @ pw[0].T.astype(np.float32)) * (cap["xp"] @ pw[1].T.astype(np.float32))).astype(np.float32))
        if rcf:
            feats.append(cap["z"])
        A = np.concatenate(feats, 1); W, lam = cv_ridge_g(grams(A, Y), LAMS if rcf else (1e-4,)); del A, feats, cap, X, Y
        Dk = W[:K].T; i = K; Cp = Ca = None
        if pw is not None:
            Cp = W[i:i + pw[0].shape[0]]; i += pw[0].shape[0]
        if rcf:
            Ca = W[i:]
        # 2) window fit. Teacher first (original layer L .. L+w on the drifted input), then install the student.
        hin, hp = grab_blocks(model, fitb, L, pw)
        upto = min(L + a.window, len(layers) - 1); mask = create_attention_mask(mx.array(hin[:1]), None)
        tgts = []
        for j in range(0, hin.shape[0], 4):
            out = mx.array(hin[j:j + 4])
            for l in range(L, upto + 1):
                out = layers[l](out, mask)
            tgts.append(np.array(out))
        tgt_all = np.concatenate(tgts, 0); del tgts
        orig_mlp = SwiGLULayer(G, U, D); o_op = lay.self_attn.o_proj
        Wo = np.array(o_op.weight, dtype=np.float64)
        st = Student(KeptMLP(G[keep], U[keep], Dk, orig_mlp), FoldO(Wo, Ca.T) if rcf else None, Cp.T if Cp is not None else None)
        lay.mlp = st.mlp
        if rcf:
            lay.self_attn.o_proj = st.o
        pos = (mx.arange(hin.shape[1]) > 0)[None, :, None]

        def run(stu, h, hpb):
            out = h
            if hpb is not None:
                out = out + (hpb @ stu.Cp.T) * pos
            for l in range(L, upto + 1):
                out = layers[l](out, mask)
            return out

        def loss_fn(stu, h, hpb, tg):
            r = (run(stu, h, hpb) - tg)[:, 1:]; c = tg[:, 1:] - tg[:, 1:].mean(-1, keepdims=True)
            return (r * r).sum() / (c * c).sum()

        model.freeze(); st.unfreeze(); st.mlp.freeze(keys=["G", "U"]); st.mlp.orig.freeze()
        if rcf:
            st.o.freeze(keys=["W0"])
        vg = nn.value_and_grad(st, loss_fn); opt = optim.Adam(learning_rate=a.lr); rng = np.random.default_rng(L)
        ts, steps, l0, lv = time.time(), 0, float('nan'), float('nan')
        while time.time() - ts < a.minutes * 60:
            idx = rng.integers(0, hin.shape[0], 4)
            lv, g = vg(st, mx.array(hin[idx]), mx.array(hp[idx]) if hp is not None else None, mx.array(tgt_all[idx]))
            opt.update(st, g); mx.eval(st.parameters(), opt.state); steps += 1
            l0 = float(lv) if steps == 1 else l0
        model.unfreeze()
        # 3) install as plain modules (same form as a1 fold placement)
        Dk = np.array(st.mlp.D, dtype=np.float64); kept[L] = (G[keep], U[keep], Dk)
        lay.mlp = SinkGuard(SwiGLULayer(*kept[L]), orig_mlp)
        if rcf:
            lay.self_attn.o_proj = SinkGuard(Lin(Wo + np.array(st.o.C, dtype=np.float64)), Lin(Wo))
            if Cp is not None:
                Gp, Up, Dp = pw; Dnew = Dp + np.array(st.Cp, dtype=np.float64)
                layers[L - 1].mlp = SinkGuard(SwiGLULayer(Gp, Up, Dnew), SwiGLULayer(*origw[L - 1]))
                if (L - 1) in kept:
                    kept[L - 1] = (Gp, Up, Dnew)
        del hin, hp, tgt_all
        log.append({"L": L, "rcf": rcf, "lam": lam, "steps": steps, "loss_first": l0, "loss_last": float(lv)})
        print(f"  L{L} {'rcf' if rcf else 'cut'} lam {lam} steps {steps} loss {l0:.4f} -> {float(lv):.4f} [{time.time()-t0:.0f}s]", flush=True)
    ce_p, ce_c = block_ce(model, dev), block_ce(model, devc)
    row = {"frac": a.frac, "K": K, "arm": a.arm, "layers": a.layers, "fold_on": a.fold_on, "minutes": a.minutes, "window": a.window,
           "dce_prose": float((ce_p - base_p).mean()), "dce_code": float((ce_c - base_c).mean()),
           "ce_prose_blocks": ce_p.tolist(), "ce_code_blocks": ce_c.tolist(), "log": log}
    path = os.path.join(ROOT, "results", "clean-a2.json"); out = json.load(open(path)) if os.path.exists(path) else {}
    key = f"{a.arm}_f{a.frac}_L{a.layers}{'_on' + a.fold_on if a.fold_on else ''}_w{a.window}_m{a.minutes}{a.tag}"
    out[key] = row; json.dump(out, open(path, "w"), indent=1)
    print(key, "dCE prose", round(row["dce_prose"], 4), "code", round(row["dce_code"], 4), f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
