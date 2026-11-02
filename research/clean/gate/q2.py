"""Q2: zero-storage fixes for the fold disturbing its own layer (CHILDREN-LOG "RCF SCALE + CHANNELS", Q2).
Single layer, fold placement (no extra storage). Arms:
  fold    closed-form RCF (as q1)
  f3_1    consistent re-fit: install, recapture layer L's real inputs, solve [hk', hp, z'] -> D h + Wo (z - z')
          (the residual after layer L then matches the original); f3_2 = one more round
  f1      short Adam fit of Dk, Cp, Ca through the real layer L (window 0), from the closed form

  python3 research/clean/gate/q2.py --layers 12,16,22,27 --Ks 48,384,768 --secs 15
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx_lm.models.base import create_attention_mask
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from contract import SwiGLULayer  # noqa: E402
from tree_lib import load_model  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import silu, boot  # noqa: E402
from g12 import capture, dev_sets, mixed_cal  # noqa: E402
from a0 import grams, cv_ridge_g, LAMS, FORBID, ROOT  # noqa: E402
from a2 import KeptMLP, FoldO, Student, grab_blocks  # noqa: E402
from q1 import setup, Installer  # noqa: E402


def f3_round(model, cal, s, keep, ins, W, K, P):
    """Install (fold placement), recapture, consistent solve. Returns new W (rows: kept, L-1 routes, attention)."""
    L = ins.L; lay = model.model.layers[L]
    ins.install(s, keep, W[:K].T, W[K:K + P], W[K + P:], "fold")
    cap = capture(model, cal, {"x": (lay, "mlp"), "z": (lay.self_attn, "o_proj")}); ins.restore()
    G, U, _ = s["W"]; Gk, Uk = G[keep].astype(np.float32), U[keep].astype(np.float32)
    Hk2 = (silu(cap["x"] @ Gk.T) * (cap["x"] @ Uk.T)).astype(np.float32)
    Wo = np.array(ins.saved[1].weight, dtype=np.float32)
    T = (s["Y"] + (s["Z"] - cap["z"]) @ Wo.T).astype(np.float32)
    A = np.concatenate([Hk2, s["Hp"], cap["z"]], 1); W2, _ = cv_ridge_g(grams(A, T), LAMS)
    return W2


def f1_fit(model, fitb, s, keep, ins, W, K, P, secs, lr=3e-4, routes=True, window=0):
    L = ins.L; layers = model.model.layers; G, U, D = s["W"]; Gp, Up, Dp = s["Wp"]
    if not routes:   # cut control: kept writes only
        return f1_cut(model, fitb, s, keep, ins, W, K, secs, lr, window)
    hin, hp = grab_blocks(model, fitb, L, (Gp, Up))
    mask = create_attention_mask(mx.array(hin[:1]), None)
    up = min(L + window, len(layers) - 1)

    def thru(h):
        for l in range(L, up + 1):
            h = layers[l](h, mask)
        return h
    tg = np.concatenate([np.array(thru(mx.array(hin[j:j + 4]))) for j in range(0, hin.shape[0], 4)], 0)
    Wo = np.array(ins.saved[1].weight, dtype=np.float64)
    st = Student(KeptMLP(G[keep], U[keep], W[:K].T, SwiGLULayer(G, U, D)), FoldO(Wo, W[K + P:].T), W[K:K + P].T)
    layers[L].mlp, layers[L].self_attn.o_proj = st.mlp, st.o
    pos = (mx.arange(hin.shape[1]) > 0)[None, :, None]

    def loss_fn(stu, h, hpb, t):
        r = (thru(h + (hpb @ stu.Cp.T) * pos) - t)[:, 1:]; c = t[:, 1:] - t[:, 1:].mean(-1, keepdims=True)
        return (r * r).sum() / (c * c).sum()

    model.freeze(); st.unfreeze(); st.mlp.freeze(keys=["G", "U"]); st.mlp.orig.freeze(); st.o.freeze(keys=["W0"])
    vg = nn.value_and_grad(st, loss_fn); opt = optim.Adam(learning_rate=lr); rng = np.random.default_rng(L); t0 = time.time(); n = 0
    while time.time() - t0 < secs:
        idx = rng.integers(0, hin.shape[0], 4)
        _, g = vg(st, mx.array(hin[idx]), mx.array(hp[idx]), mx.array(tg[idx])); opt.update(st, g); mx.eval(st.parameters(), opt.state); n += 1
    model.unfreeze(); ins.restore()
    return np.concatenate([np.array(st.mlp.D, dtype=np.float64).T, np.array(st.Cp, dtype=np.float64).T, np.array(st.o.C, dtype=np.float64).T], 0), n


def f1_cut(model, fitb, s, keep, ins, Wc, K, secs, lr, window=0):
    L = ins.L; layers = model.model.layers; G, U, D = s["W"]
    hin, _ = grab_blocks(model, fitb, L, None)
    mask = create_attention_mask(mx.array(hin[:1]), None)
    up = min(L + window, len(layers) - 1)

    def thru(h):
        for l in range(L, up + 1):
            h = layers[l](h, mask)
        return h
    tg = np.concatenate([np.array(thru(mx.array(hin[j:j + 4]))) for j in range(0, hin.shape[0], 4)], 0)
    st = Student(KeptMLP(G[keep], U[keep], Wc[:K].T, SwiGLULayer(G, U, D))); layers[L].mlp = st.mlp

    def loss_fn(stu, h, t):
        r = (thru(h) - t)[:, 1:]; c = t[:, 1:] - t[:, 1:].mean(-1, keepdims=True)
        return (r * r).sum() / (c * c).sum()

    model.freeze(); st.unfreeze(); st.mlp.freeze(keys=["G", "U"]); st.mlp.orig.freeze()
    vg = nn.value_and_grad(st, loss_fn); opt = optim.Adam(learning_rate=lr); rng = np.random.default_rng(L); t0 = time.time(); n = 0
    while time.time() - t0 < secs:
        idx = rng.integers(0, hin.shape[0], 4)
        _, g = vg(st, mx.array(hin[idx]), mx.array(tg[idx])); opt.update(st, g); mx.eval(st.parameters(), opt.state); n += 1
    model.unfreeze(); ins.restore()
    return np.array(st.mlp.D, dtype=np.float64).T, n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="12,16,22,27"); ap.add_argument("--Ks", default="48,384,768")
    ap.add_argument("--ncal", type=int, default=128); ap.add_argument("--secs", type=float, default=15); ap.add_argument("--nfit", type=int, default=64)
    ap.add_argument("--out", default="clean-q2.json"); ap.add_argument("--q4", action="store_true"); ap.add_argument("--window", type=int, default=0)
    a = ap.parse_args(); layers = [int(x) for x in a.layers.split(",")]; assert not FORBID & set(layers)
    model, tok = load_model(); cal = mixed_cal(tok, a.ncal); dev, devc = dev_sets(tok)
    fitb = mx.concatenate([cal[:a.nfit], cal[a.ncal:a.ncal + a.nfit]], 0)
    texts = (("prose", dev), ("code", devc)); base = {t: block_ce(model, b) for t, b in texts}
    path = os.path.join(ROOT, "results", a.out); out = json.load(open(path)) if os.path.exists(path) else {}; t0 = time.time()
    for L in layers:
        s = setup(model, L, cal); ins = Installer(model, L); R = np.concatenate([s["Hp"], s["Z"]], 1); P = s["Hp"].shape[1]
        for K in [int(k) for k in a.Ks.split(",")]:
            keep = np.sort(np.argsort(-s["en"])[:K]); Hk = s["H"][:, keep]
            Wc, _ = cv_ridge_g(grams(Hk, s["Y"]), (1e-4,))
            A = np.concatenate([Hk, R], 1); W, _ = cv_ridge_g(grams(A, s["Y"]), LAMS); del A
            Ws = {"fold": W}
            if not a.q4:
                Ws["f3_1"] = f3_round(model, cal, s, keep, ins, W, K, P)
                Ws["f3_2"] = f3_round(model, cal, s, keep, ins, Ws["f3_1"], K, P)
            Ws["f1"], steps = f1_fit(model, fitb, s, keep, ins, W, K, P, a.secs, window=a.window)
            res = {}
            ins.install(s, keep, Wc.T, None, None, "fold"); res["cut"] = {t: block_ce(model, b) for t, b in texts}; ins.restore()
            if a.q4:
                Wcf, _ = f1_fit(model, fitb, s, keep, ins, Wc, K, P, a.secs, routes=False, window=a.window)
                ins.install(s, keep, Wcf.T, None, None, "fold"); res["cut_f1"] = {t: block_ce(model, b) for t, b in texts}; ins.restore()
            for arm, Wa in Ws.items():
                ins.install(s, keep, Wa[:K].T, Wa[K:K + P], Wa[K + P:], "fold"); res[arm] = {t: block_ce(model, b) for t, b in texts}; ins.restore()
            row = {"f1_steps": steps}
            for t, _ in texts:
                c = float((res["cut"][t] - base[t]).mean()); row[t] = {"cut": c}
                for arm in list(Ws) + (["cut_f1"] if a.q4 else []):
                    v = float((res[arm][t] - base[t]).mean()); row[t][arm] = {"dce": v, "share": 1 - v / c, "minus_cut": boot(res[arm][t] - res["cut"][t])}
            out[f"L{L}_K{K}"] = row; json.dump(out, open(path, "w"), indent=1)
            print(f"L{L} K{K} " + " | ".join(f"{t}: " + " ".join(f"{arm} {row[t][arm]['share']:.0%}" for arm in list(Ws) + (["cut_f1"] if a.q4 else [])) for t, _ in texts)
                  + f" [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
