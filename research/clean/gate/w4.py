"""W4: best routes from ANY earlier layer (CHILDREN-LOG W4). Local only (no dCE). Rows: odd positions only (memory).
Base = kept hiddens + L-1 hiddens + L attention outputs (RCF today). Leftover E = removed output - base fit.
Every MLP neuron of layers 1..L-2 is scored on calibration rows by the leftover energy it explains alone
(|h_j . E|^2 / |h_j|^2). Arms add neurons to the base and report the held-out (dev rows) "neither" share:
  l2_full      all 1536 neurons of layer L-2 (= W3 +l2)
  l2_top512    best 512 of layer L-2
  any_top512 / any_top1536 / any_top4096   best neurons from ANY of layers 1..L-2 (global ranking)

  python3 research/clean/gate/w4.py --layers 16,22 --Ks 48,768
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx, mlx.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from a0 import grams, cv_ridge_g, LAMS, FORBID, ROOT  # noqa: E402
from contract import orig_weights  # noqa: E402
from g6 import silu  # noqa: E402
from g12 import dev_sets, mixed_cal  # noqa: E402
from tree_lib import load_model  # noqa: E402


class OddGrab(nn.Module):
    def __init__(self, inner, store, key):
        super().__init__(); self.inner, self.store, self.key = inner, store, key
    def __call__(self, x):
        self.store.setdefault(self.key, []).append(np.array(x[:, 1::2]).reshape(-1, x.shape[-1]).astype(np.float32)); return self.inner(x)


def grab_all(model, blocks, L, batch=4):
    """MLP inputs of layers 1..L and L's o_proj input, odd positions only."""
    m = model.model.layers; st = {}; saved = {}
    for l in range(1, L + 1):
        saved[l] = m[l].mlp; m[l].mlp = OddGrab(m[l].mlp, st, l)
    so = m[L].self_attn.o_proj; m[L].self_attn.o_proj = OddGrab(so, st, "z")
    for i in range(0, blocks.shape[0], batch):
        mx.eval(model(blocks[i:i + batch, :-1]))
    for l in saved:
        m[l].mlp = saved[l]
    m[L].self_attn.o_proj = so
    return {k: np.concatenate(v, 0) for k, v in st.items()}


def hid(model, X, l):
    G, U, _ = orig_weights(model, l); return (silu(X @ G.T.astype(np.float32)) * (X @ U.T.astype(np.float32))).astype(np.float32)


def neither(A, r, Ad, rd):
    W, _ = cv_ridge_g(grams(A, r), LAMS); P = Ad @ W.astype(np.float32)
    return float(np.sum((rd - P) ** 2) / np.sum((rd - rd.mean(0)) ** 2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="16,22"); ap.add_argument("--Ks", default="48,768")
    a = ap.parse_args(); layers = [int(x) for x in a.layers.split(",")]; assert not FORBID & set(layers)
    model, tok = load_model(); cal = mixed_cal(tok, 128); dev, devc = dev_sets(tok); devall = mx.concatenate([dev, devc], 0)
    path = os.path.join(ROOT, "results", "clean-w4.json"); out = json.load(open(path)) if os.path.exists(path) else {}; t0 = time.time()
    for L in layers:
        c, d = grab_all(model, cal, L), grab_all(model, devall, L)
        H, Hd = hid(model, c[L], L), hid(model, d[L], L); D = orig_weights(model, L)[2].astype(np.float32)
        base_c = [hid(model, c[L - 1], L - 1), c["z"]]; base_d = [hid(model, d[L - 1], L - 1), d["z"]]
        en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0); order = np.argsort(-en)
        print(f"L{L} captured [{time.time()-t0:.0f}s]", flush=True)
        for K in [int(k) for k in a.Ks.split(",")]:
            keep, rem = np.sort(order[:K]), np.sort(order[K:])
            r, rd = H[:, rem] @ D[:, rem].T, Hd[:, rem] @ D[:, rem].T
            A = np.concatenate([H[:, keep]] + base_c, 1); Ad = np.concatenate([Hd[:, keep]] + base_d, 1)
            W, _ = cv_ridge_g(grams(A, r), LAMS); E = r - A @ W.astype(np.float32)
            row = {"base": neither(A, r, Ad, rd)}
            # score every neuron of layers 1..L-2 on calibration rows
            scores, cols_c, cols_d, src = [], [], [], []
            for l in range(1, L - 1):
                h = hid(model, c[l], l); sc = np.sum((h.T @ E) ** 2, 1) / np.maximum(np.sum(h ** 2, 0), 1e-9)
                top = np.argsort(-sc)[:512]                                   # keep the 512 best per layer as candidates
                scores.append(sc[top]); cols_c.append(h[:, top]); cols_d.append(hid(model, d[l], l)[:, top]); src.append(np.full(len(top), l))
                if l == L - 2:
                    l2c, l2d = h, hid(model, d[l], l)
                del h
            scores, src = np.concatenate(scores), np.concatenate(src); Cc, Cd = np.concatenate(cols_c, 1), np.concatenate(cols_d, 1); del cols_c, cols_d
            row["l2_full"] = neither(np.concatenate([A, l2c], 1), r, np.concatenate([Ad, l2d], 1), rd)
            i2 = np.where(src == L - 2)[0][:512]
            row["l2_top512"] = neither(np.concatenate([A, Cc[:, i2]], 1), r, np.concatenate([Ad, Cd[:, i2]], 1), rd)
            g = np.argsort(-scores)
            for n in (512, 1536, 4096):
                idx = g[:n]
                row[f"any_top{n}"] = neither(np.concatenate([A, Cc[:, idx]], 1), r, np.concatenate([Ad, Cd[:, idx]], 1), rd)
                row[f"any_top{n}_layers"] = {int(l): int(np.sum(src[idx] == l)) for l in np.unique(src[idx])}
            out[f"L{L}_K{K}"] = row; json.dump(out, open(path, "w"), indent=1)
            print(f"L{L} K{K} neither: " + " ".join(f"{k} {v:.3f}" for k, v in row.items() if not isinstance(v, dict)) + f" [{time.time()-t0:.0f}s]", flush=True)
            print(f"   top1536 from layers: {row['any_top1536_layers']}", flush=True)
            del Cc, Cd, l2c, l2d, A, Ad
        del c, d, H, Hd


if __name__ == "__main__":
    main()
