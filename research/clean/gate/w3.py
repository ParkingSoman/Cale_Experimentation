"""W3: what can reach the "neither" part of the removed output? (CHILDREN-LOG W3). Local only (no dCE).
Held-out (dev rows) R^2 of the removed output r from feature sets, fit on calibration rows:
  base   = kept hiddens + routes (L-1 hiddens, L attention head outputs)       [RCF today]
  +l2    = base + layer L-2 hiddens                                          [fold into L-2 write: zero storage]
  +z1    = base + layer L-1 attention head outputs                           [fold into L-1 o_proj: zero storage]
  +x     = base + the MLP input x itself (linear)                            [costs a 576x576 map]
  +all   = base + l2 + z1 + x

  python3 research/clean/gate/w3.py --layers 16,22 --Ks 48,768
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from a0 import grams, cv_ridge_g, LAMS, FORBID, ROOT  # noqa: E402
from contract import orig_weights  # noqa: E402
from g6 import silu  # noqa: E402
from g12 import capture, dev_sets, mixed_cal  # noqa: E402
from tree_lib import load_model  # noqa: E402
import mlx.core as mx  # noqa: E402


def feats(model, blocks, L):
    m = model.model.layers
    cap = capture(model, blocks, {"x": (m[L], "mlp"), "z": (m[L].self_attn, "o_proj"), "xp": (m[L - 1], "mlp"),
                                  "xp2": (m[L - 2], "mlp"), "z1": (m[L - 1].self_attn, "o_proj")})
    def hid(X, l):
        G, U, _ = orig_weights(model, l); return (silu(X @ G.T.astype(np.float32)) * (X @ U.T.astype(np.float32))).astype(np.float32)
    G, U, D = orig_weights(model, L)
    return dict(x=cap["x"], H=hid(cap["x"], L), Hp=hid(cap["xp"], L - 1), Hp2=hid(cap["xp2"], L - 2), z=cap["z"], z1=cap["z1"], D=D.astype(np.float32))


def r2(Y, P):
    return float(1 - np.sum((Y - P) ** 2) / np.sum((Y - Y.mean(0)) ** 2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="16,22"); ap.add_argument("--Ks", default="48,768")
    a = ap.parse_args(); layers = [int(x) for x in a.layers.split(",")]; assert not FORBID & set(layers)
    model, tok = load_model(); cal = mixed_cal(tok, 128); dev, devc = dev_sets(tok); devall = mx.concatenate([dev, devc], 0)
    path = os.path.join(ROOT, "results", "clean-w3.json"); out = json.load(open(path)) if os.path.exists(path) else {}; t0 = time.time()
    for L in layers:
        f, fd = feats(model, cal, L), feats(model, devall, L)
        en = np.linalg.norm(f["H"], axis=0) * np.linalg.norm(f["D"], axis=0); order = np.argsort(-en)
        for K in [int(k) for k in a.Ks.split(",")]:
            keep, rem = np.sort(order[:K]), np.sort(order[K:])
            r, rd = f["H"][:, rem] @ f["D"][:, rem].T, fd["H"][:, rem] @ fd["D"][:, rem].T
            parts = {"base": ["Hk", "Hp", "z"], "+l2": ["Hk", "Hp", "z", "Hp2"], "+z1": ["Hk", "Hp", "z", "z1"],
                     "+x": ["Hk", "Hp", "z", "x"], "+all": ["Hk", "Hp", "z", "Hp2", "z1", "x"]}
            get = lambda F, k: F["H"][:, keep] if k == "Hk" else F[k]
            row = {}
            for nm, ks in parts.items():
                A = np.concatenate([get(f, k) for k in ks], 1); Ad = np.concatenate([get(fd, k) for k in ks], 1)
                W, _ = cv_ridge_g(grams(A, r), LAMS); row[nm] = 1 - r2(rd, Ad @ W.astype(np.float32)); del A, Ad
            out[f"L{L}_K{K}"] = row; json.dump(out, open(path, "w"), indent=1)
            print(f"L{L} K{K} 'neither' (unreached share of removed output): " + " ".join(f"{k} {v:.3f}" for k, v in row.items()) + f" [{time.time()-t0:.0f}s]", flush=True)
        del f, fd


if __name__ == "__main__":
    main()
