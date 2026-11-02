"""Q1: single-layer fold vs oracle placement across widths (CHILDREN-LOG "RCF SCALE + CHANNELS", Q1).
One RCF fit per (layer, K). Arms differ only in WHERE the route parts are written:
  fold      L-1 fold in layer L-1's write matrix, attention fold in layer L's o_proj (rcf_v1 / a0)
  oracle    both written after MLP L (layer L never reads them; extra storage, upper bound)
  oracle_p  only the L-1 fold moved after MLP L
  oracle_a  only the attention fold moved after MLP L

  python3 research/clean/gate/q1.py --layers 12,16,22,27 --Ks 48,384,768
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from contract import orig_weights, ROOT, SwiGLULayer  # noqa: E402
from tree_lib import load_model  # noqa: E402
from r1 import block_ce  # noqa: E402
from g6 import silu, boot  # noqa: E402
from g10 import Lin  # noqa: E402
from g12 import capture, dev_sets, mixed_cal  # noqa: E402
from sink import SinkGuard  # noqa: E402
from a0 import grams, cv_ridge_g, LAMS, FORBID  # noqa: E402
from a1 import Stash, ZStash, AddAfter  # noqa: E402

ARMS = ("fold", "oracle", "oracle_p", "oracle_a")


def setup(model, L, ncal_blocks):
    """Captured features and targets for one layer (as a0)."""
    lay = model.model.layers[L]
    cap = capture(model, ncal_blocks, {"x": (lay, "mlp"), "z": (lay.self_attn, "o_proj"), "xp": (model.model.layers[L - 1], "mlp")})
    G, U, D = orig_weights(model, L); X = cap["x"]
    H = (silu(X @ G.T.astype(np.float32)) * (X @ U.T.astype(np.float32))).astype(np.float32)
    Y = (H @ D.T.astype(np.float32)).astype(np.float32)
    en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0)
    Gp, Up, Dp = orig_weights(model, L - 1); Xp = cap["xp"]
    Hp = (silu(Xp @ Gp.T.astype(np.float32)) * (Xp @ Up.T.astype(np.float32))).astype(np.float32)
    return dict(X=X, H=H, Y=Y, en=en, Hp=Hp, Z=cap["z"], W=(G, U, D), Wp=(Gp, Up, Dp))


class Installer:
    """Installs a coarse layer L plus route parts in a chosen placement; restore() puts the original modules back."""
    def __init__(self, model, L):
        self.m, self.L = model.model.layers, L
        self.saved = (self.m[L].mlp, self.m[L].self_attn.o_proj, self.m[L - 1].mlp)

    def install(self, s, keep, Dk, Cp, Ca, place):
        L, m = self.L, self.m; G, U, D = s["W"]; Gp, Up, Dp = s["Wp"]; box = {}
        Wo = np.array(self.saved[1].weight, dtype=np.float64)
        mlp = SinkGuard(SwiGLULayer(G[keep], U[keep], Dk), SwiGLULayer(G, U, D)); parts = []
        if Cp is not None:
            if place in ("oracle", "oracle_p"):
                m[L - 1].mlp = Stash(self.saved[2], Gp, Up, box, "p"); parts.append(("p", Cp.T))
            else:
                m[L - 1].mlp = SinkGuard(SwiGLULayer(Gp, Up, Dp + Cp.T), SwiGLULayer(Gp, Up, Dp))
        if Ca is not None:
            if place in ("oracle", "oracle_a"):
                m[L].self_attn.o_proj = ZStash(self.saved[1], box, "z"); parts.append(("z", Ca.T))
            else:
                m[L].self_attn.o_proj = SinkGuard(Lin(Wo + Ca.T), Lin(Wo))
        m[L].mlp = AddAfter(mlp, parts, box) if parts else mlp

    def restore(self):
        self.m[self.L].mlp, self.m[self.L].self_attn.o_proj, self.m[self.L - 1].mlp = self.saved


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="12,16,22,27"); ap.add_argument("--Ks", default="48,384,768")
    ap.add_argument("--ncal", type=int, default=128); ap.add_argument("--arms", default=",".join(ARMS)); ap.add_argument("--out", default="clean-q1.json")
    a = ap.parse_args(); layers = [int(x) for x in a.layers.split(",")]; assert not FORBID & set(layers)
    model, tok = load_model(); cal = mixed_cal(tok, a.ncal); dev, devc = dev_sets(tok)
    texts = (("prose", dev), ("code", devc)); base = {t: block_ce(model, b) for t, b in texts}
    path = os.path.join(ROOT, "results", a.out); out = json.load(open(path)) if os.path.exists(path) else {}; t0 = time.time()
    for L in layers:
        s = setup(model, L, cal); ins = Installer(model, L); R = np.concatenate([s["Hp"], s["Z"]], 1); P = s["Hp"].shape[1]
        for K in [int(k) for k in a.Ks.split(",")]:
            keep = np.sort(np.argsort(-s["en"])[:K]); Hk = s["H"][:, keep]
            Wc, _ = cv_ridge_g(grams(Hk, s["Y"]), (1e-4,))
            A = np.concatenate([Hk, R], 1); W, lam = cv_ridge_g(grams(A, s["Y"]), LAMS); del A
            res = {}
            ins.install(s, keep, Wc.T, None, None, "fold"); res["cut"] = {t: block_ce(model, b) for t, b in texts}; ins.restore()
            for arm in a.arms.split(","):
                ins.install(s, keep, W[:K].T, W[K:K + P], W[K + P:], arm); res[arm] = {t: block_ce(model, b) for t, b in texts}; ins.restore()
            row = {"lam": lam}
            for t, _ in texts:
                c = float((res["cut"][t] - base[t]).mean()); row[t] = {"cut": c}
                for arm in a.arms.split(","):
                    v = float((res[arm][t] - base[t]).mean()); row[t][arm] = {"dce": v, "share": 1 - v / c, "minus_cut": boot(res[arm][t] - res["cut"][t])}
            out[f"L{L}_K{K}"] = row; json.dump(out, open(path, "w"), indent=1)
            print(f"L{L} K{K} " + " | ".join(f"{t}: " + " ".join(f"{arm} {row[t][arm]['share']:.0%}" for arm in a.arms.split(",")) for t, _ in texts)
                  + f" [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
