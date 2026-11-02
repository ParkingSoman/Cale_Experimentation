"""D5 strong training reference: drop block B, train block B+1 (the last block when B = 16) to match the intact model's
logits (KL), starting from its original weights. Train on analysis ranks 0-17 (18k images), early-stop on ranks 18-19
(2k), score the best epoch on the sealed 3k. Saves best weights to hf/imagenet/d5cache/tr_{B}_{params}_{lr}.npz.
usage: python d5_train.py B PARAMS(all|fc) LR [EPOCHS]"""
import os, sys, time, json
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten
from convnext import *
B, PAR, LR = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]); EP = int(sys.argv[4]) if len(sys.argv) > 4 else 12
t0 = time.time(); m = load_convnext(); ST = m.stage_of; D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, f"x{B}.npz"))
XA, LA, XE, LE = mx.array(z["XA"]), mx.array(z["LA"]), mx.array(z["XE"]), mx.array(z["LE"])
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
TRI, VAI = np.where(rk < 18)[0], np.where(rk >= 18)[0]
blk = m.blocks[B + 1]
def tail(x):
    x = blk(x.astype(mx.float32))
    for i in range(B + 2, 18):
        if ST[i] != ST[i - 1]: x = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](x))
        x = m.blocks[i](x)
    return m.head(x)
def kl(L, T): lp = L - mx.logsumexp(L, 1, keepdims=True); tp = T - mx.logsumexp(T, 1, keepdims=True); return (mx.exp(tp) * (tp - lp)).sum(1)
def evalkl(Xs, Ts, I=None):
    I = np.arange(Xs.shape[0]) if I is None else I; out = []
    for a in range(0, len(I), 200): j = mx.array(I[a:a + 200]); out.append(np.array(kl(tail(Xs[j]), Ts[j])))
    return float(np.concatenate(out).mean())
m.freeze(); keys = ["fc1", "fc2"] if PAR == "fc" else ["conv_dw", "norm", "fc1", "fc2", "gamma"]
blk.unfreeze(keys=keys) if PAR == "fc" else blk.unfreeze()
if PAR == "fc": blk.freeze(); blk.fc1.unfreeze(); blk.fc2.unfreeze()
opt = optim.Adam(LR); lossf = lambda mod, xb, tb: kl(tail(xb), tb).mean(); vg = nn.value_and_grad(m, lossf)
best = (evalkl(XA, LA, VAI), -1); start_sealed = evalkl(XE, LE); print(f"start: val {best[0]:.4f} sealed {start_sealed:.4f}", flush=True)
snap = {k: np.array(v) for k, v in tree_flatten(blk.parameters())}; SEED = int(os.environ.get('SEED', 0)); rng = np.random.default_rng(SEED); hist = []
for ep in range(EP):
    p = rng.permutation(TRI)
    for a in range(0, len(p), 64):
        j = mx.array(p[a:a + 64]); l, g = vg(m, XA[j], LA[j]); opt.update(m, g); mx.eval(m.parameters(), opt.state)
    v = evalkl(XA, LA, VAI); hist.append(v)
    if v < best[0]: best = (v, ep); snap = {k: np.array(v_) for k, v_ in tree_flatten(blk.parameters())}
    print(f"  epoch {ep}: val {v:.4f}  {time.time() - t0:.0f}s", flush=True)
blk.load_weights(list((k, mx.array(v)) for k, v in snap.items())); s = evalkl(XE, LE)
print(f"drop block {B}, train block {B + 1} [{PAR}, lr {LR}]: best epoch {best[1]} val {best[0]:.4f} -> SEALED KL {s:.4f} (start {start_sealed:.4f})")
np.savez(os.path.join(D, f"tr_{B}_{PAR}_{LR:g}" + (f"_s{SEED}" if SEED else "") + ".npz"), **snap)
json.dump({"B": B, "params": PAR, "lr": LR, "val_hist": hist, "best_epoch": best[1], "sealed": s, "start": start_sealed}, open(os.path.join(RES, f"vision-d5-train-{B}-{PAR}-{LR:g}" + (f"-s{SEED}" if SEED else "") + ".json"), "w"))
