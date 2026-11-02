"""TF1 tree fold: each removed layer-L neuron folds into ONE layer-(L-1) parent found by attribution.
Rules: CHILDREN-LOG.md "TREE FOLD".   python3 research/clean/gate/tf1.py --L 16 --K 48"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.join(CO, "methods")); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from contract import orig_weights, SwiGLULayer, SEQ, ROOT  # noqa: E402
from tree_lib import load_model, token_blocks  # noqa: E402
from r1 import block_ce  # noqa: E402
import rcf_v1  # noqa: E402
from rcf_v1 import SinkGuard, _capture, _cv_ridge, silu  # noqa: E402

FORBID = {6, 9, 21, 24}


def dsilu(z):
    s = 1 / (1 + np.exp(-z)); return s * (1 + z * (1 - s))


def boot(d, n=2000):
    r = np.random.default_rng(0); m = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--L", type=int); ap.add_argument("--K", type=int, default=48)
    a = ap.parse_args(); L, K = a.L, a.K; assert L not in FORBID and L >= 2
    model, tok = load_model()
    cal = token_blocks(tok, "train", 64)
    txt = open(os.path.join(ROOT, "research", "clean", "data", "code_train.txt")).read()
    ids = np.array(tok.encode(txt[400000:1600000]), dtype=np.int32)[: 64 * 513]
    calm = mx.concatenate([cal, mx.array(ids.reshape(64, 513))], 0)
    dev = token_blocks(tok, "test", 40)
    ci = np.array(tok.encode(txt[:400000]), dtype=np.int32); code = mx.array(ci[:40 * 513].reshape(40, 513))
    base = {"prose": block_ce(model, dev), "code": block_ce(model, code)}
    lay, layp = model.model.layers[L], model.model.layers[L - 1]
    cap = _capture(model, calm, {"x": (lay, "mlp"), "xp": (layp, "mlp"), "s": (lay, "post_attention_layernorm")})
    G, U, D = orig_weights(model, L); Gp, Up, Dp = orig_weights(model, L - 1)
    X = cap["x"].astype(np.float64); Xp = cap["xp"].astype(np.float64); S = cap["s"].astype(np.float64)
    Zg, Zu = X @ G.T, X @ U.T; H = silu(Zg) * Zu; Hp = silu(Xp @ Gp.T) * (Xp @ Up.T); Y = H @ D.T
    rms = np.sqrt((S ** 2).mean(1) + 1e-5); gam = np.array(lay.post_attention_layernorm.weight, np.float64)
    en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0); order = np.argsort(-en)
    keep, rem = np.sort(order[:K]), np.sort(order[K:])
    MG, MU = (G * gam) @ Dp, (U * gam) @ Dp          # (n_child, n_parent): d_p . (gamma * G_c), d_p . (gamma * U_c)
    ntop = max(1, int(.005 * len(H)))
    Hpr = Hp / rms[:, None]
    attr_par = {}; pred_par = {}
    Hz = (Hp - Hp.mean(0)) / (Hp.std(0) + 1e-12)
    for c in rem:
        t = np.argpartition(-np.abs(H[:, c]), ntop)[:ntop]
        A = (dsilu(Zg[t, c]) * Zu[t, c])[:, None] * Hpr[t]; B = silu(Zg[t, c])[:, None] * Hpr[t]
        at = MG[c] * A.mean(0) + MU[c] * B.mean(0)
        attr_par[c] = np.argsort(-np.abs(at))[:3]
        hc = (H[:, c] - H[:, c].mean()) / (H[:, c].std() + 1e-12)
        pred_par[c] = np.array([np.argmax(np.abs(hc @ Hz))])

    def tree_fold(par, n):
        """alpha per child on its first n parents; returns Acp (n_rem x n_parent) and predicted removed output."""
        Acp = np.zeros((len(rem), Hp.shape[1]))
        for i, c in enumerate(rem):
            ps = par[c][:n]; F = Hp[:, ps]
            Acp[i, ps] = np.linalg.lstsq(F, H[:, c], rcond=None)[0]
        return Acp, (Hp @ Acp.T) @ D[:, rem].T

    def ls(Hk, T):
        M = Hk.T @ Hk; M += np.eye(len(M)) * 1e-4 * np.trace(M) / len(M); return np.linalg.solve(M, Hk.T @ T).T

    arms = {}
    Rrem = H[:, rem] @ D[:, rem].T
    arms["cut_ls"] = (ls(H[:, keep], Y), None)
    arms["cut_raw"] = (D[:, keep], None)
    for name, par, n in (("tree1", attr_par, 1), ("tree3", attr_par, 3), ("tree1_pred", pred_par, 1)):
        Acp, Rhat = tree_fold(par, n); Dpn = Dp + D[:, rem] @ Acp
        arms[name] = (ls(H[:, keep], Y - Rhat), Dpn)
        if name != "tree1_pred":
            arms[name + "_raw"] = (D[:, keep], Dpn)
        print(name, "fold R^2 on removed output %.3f" % (1 - ((Rrem - Rhat) ** 2).sum() / (Rrem ** 2).sum()), flush=True)
    W = _cv_ridge(np.concatenate([H[:, keep], Hp], 1).astype(np.float32), Y.astype(np.float32))
    arms["rcf_plf"] = (W[:K].T, Dp + W[K:].T)
    Cp = _cv_ridge(Hp.astype(np.float32), Rrem.astype(np.float32))
    arms["rcf_plf_raw"] = (D[:, keep], Dp + Cp.T)
    out = {"L": L, "K": K, "arms": {}}
    per = {}
    origL, origP, origO = lay.mlp, layp.mlp, lay.self_attn.o_proj
    for name, (Dk, Dpn) in list(arms.items()) + [("rcf_v1", (None, None))]:
        if name == "rcf_v1":
            new = rcf_v1.build(model, tok, L, K, cal, log=lambda *x: None); extra = new.extra
        else:
            new = SinkGuard(SwiGLULayer(G[keep], U[keep], Dk), SwiGLULayer(G, U, D)); extra = []
            if Dpn is not None:
                extra = [(layp, "mlp", SinkGuard(SwiGLULayer(Gp, Up, Dpn), SwiGLULayer(Gp, Up, Dp)))]
        lay.mlp = new
        for o, at_, m in extra: setattr(o, at_, m)
        per[name] = {t: block_ce(model, b) - base[t] for t, b in (("prose", dev), ("code", code))}
        lay.mlp, layp.mlp, lay.self_attn.o_proj = origL, origP, origO
        out["arms"][name] = {t: boot(v) for t, v in per[name].items()}
        print(name, {t: "%.4f" % v[0] for t, v in out["arms"][name].items()}, flush=True)
    for reg, cut in (("refit", "cut_ls"), ("raw", "cut_raw")):
        for n in per:
            if (reg == "raw") != n.endswith("_raw") or n == cut: continue
            out.setdefault("share", {})[n] = {t: float((per[cut][t].mean() - per[n][t].mean()) / per[cut][t].mean()) for t in ("prose", "code")}
    s = out["share"]
    out["tree1_over_plf"] = {t: s["tree1"][t] / s["rcf_plf"][t] if s["rcf_plf"][t] > 0 else None for t in ("prose", "code")}
    print("shares", json.dumps(s)); print("tree1 / rcf_plf", out["tree1_over_plf"])
    p = os.path.join(ROOT, "results", "clean-tf1.json"); res = json.load(open(p)) if os.path.exists(p) else {}
    res[f"L{L}_K{K}"] = out; json.dump(res, open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
