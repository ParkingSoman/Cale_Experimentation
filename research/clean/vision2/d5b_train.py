"""D5b stronger training reference for "drop block DROP". Start = the stream entering block START (cache x{START}.npz);
run blocks START..17 except DROP, then the head. Train the blocks in TRAIN (comma list; 'head' allowed) to the intact
logits (KL). Options (env): BS batch (64), LR, EP epochs, COS=1 cosine decay with 1-epoch warmup, SMALLFREEZE=1 freezes
gamma / LN / biases (tiny-scale params), SEED. Logs train KL (2k train images) and val KL each epoch; reports the number
of trainable weights; scores the best-val snapshot on the sealed 3k.
usage: python d5b_train.py START DROP TRAIN"""
import os, sys, time, json
import numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_flatten
from convnext import *
START, DROP, TRAIN = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3].split(",")
E = lambda k, d: type(d)(os.environ.get(k, d)); BS, LR, EP, COS, SF, SEED = E("BS", 64), E("LR", 1e-4), E("EP", 12), E("COS", 0), E("SMALLFREEZE", 0), E("SEED", 0)
MORE, FEAT, LORA = E("MORE", 0), E("FEAT", 0.0), E("LORA", 0)
t0 = time.time(); m = load_convnext(); ST = m.stage_of; D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, f"x{START}.npz"))
XA, LA, XE, LE = mx.array(z["XA"]), mx.array(z["LA"]), mx.array(z["XE"]), mx.array(z["LE"])
rk = np.zeros(len(z["ana"]), int)
for c in range(1000): ix = np.where(z["yA"] == c)[0]; rk[ix] = np.arange(len(ix))
TRI, VAI = np.where(rk < 18)[0], np.where(rk >= 18)[0]
if MORE:
    zb = np.load(os.path.join(D, f'x{START}_more.npz')); n0 = XA.shape[0]; XA = mx.concatenate([XA, mx.array(zb['XB'])]); LA = mx.concatenate([LA, mx.array(zb['LB'])]); TRI = np.r_[TRI, n0 + np.arange(zb['XB'].shape[0])]
TRS = np.random.default_rng(1).choice(TRI, 2000, replace=False)
BL = [b for b in range(START, 18) if b != DROP]
def stream(x):
    x = x.astype(mx.float32)
    for i in BL:
        if i > START and ST[i] != ST[i - 1]: x = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](x))
        x = m.blocks[i](x)
    return x
def tail(x): return m.head(stream(x))
def kl(L, T): lp = L - mx.logsumexp(L, 1, keepdims=True); tp = T - mx.logsumexp(T, 1, keepdims=True); return (mx.exp(tp) * (tp - lp)).sum(1)
def evalkl(Xs, Ts, I=None):
    I = np.arange(Xs.shape[0]) if I is None else I; out = []
    for a in range(0, len(I), 200): j = mx.array(I[a:a + 200]); out.append(np.array(kl(tail(Xs[j]), Ts[j])))
    return float(np.concatenate(out).mean())
class LoRA(nn.Module):
    def __init__(s, base, r):
        super().__init__(); s.base = base; o, i = base.weight.shape; s.A = mx.random.normal((o, r)) * 0.0; s.Bm = mx.random.normal((r, i)) * (1 / i ** .5)
    def __call__(s, x): return s.base(x) + (x @ s.Bm.T) @ s.A.T
import copy; ORIG = {b: copy.deepcopy(m.blocks[b]) for b in range(START, 18)}
def intact_stream(x):
    x = x.astype(mx.float32)
    for i in range(START, 18):
        if i > START and ST[i] != ST[i - 1]: x = m.down[ST[i] - 1](m.down_norm[ST[i] - 1](x))
        x = ORIG[i](x)
    return x
m.freeze(); mods = []
for t in TRAIN:
    if t == "head": mods += [m.head_norm, m.fc]
    else:
        b = m.blocks[int(t)]
        if LORA: b.fc1 = LoRA(b.fc1, LORA); b.fc2 = LoRA(b.fc2, LORA); b.fc1.base.freeze(); b.fc2.base.freeze()
        elif SF: b.fc1.unfreeze(keys=["weight"]); b.fc2.unfreeze(keys=["weight"]); b.conv_dw.unfreeze(keys=["weight"])
        else: b.unfreeze()
        mods.append(b)
    if t == "head": m.head_norm.unfreeze(); m.fc.unfreeze()
ntr = sum(v.size for _, v in tree_flatten(m.trainable_parameters())); print(f"trainable weights: {ntr:,}; blocks run {BL}; train {TRAIN}; BS {BS} LR {LR} EP {EP} COS {COS} SMALLFREEZE {SF}", flush=True)
steps = EP * (len(TRI) // BS)
sched = optim.join_schedules([optim.linear_schedule(LR / 20, LR, len(TRI) // BS), optim.cosine_decay(LR, steps)], [len(TRI) // BS]) if COS else LR
VT = float(((intact_stream(XA[mx.array(TRS[:200])]) - intact_stream(XA[mx.array(TRS[:200])]).mean((0, 1, 2))) ** 2).mean()) if FEAT else 1.0
opt = optim.Adam(sched)
def lossf(mod, xb, tb):
    S = stream(xb); l = kl(m.head(S), tb).mean()
    if FEAT: l = l + FEAT * ((S - mx.stop_gradient(intact_stream(xb))) ** 2).mean() / VT
    return l
vg = nn.value_and_grad(m, lossf)
snapf = lambda: {k: np.array(v) for k, v in tree_flatten(m.trainable_parameters())}
best = (evalkl(XA, LA, VAI), -1); start_sealed = evalkl(XE, LE); snap = snapf()
print(f"start: train {evalkl(XA, LA, TRS):.4f} val {best[0]:.4f} sealed {start_sealed:.4f}", flush=True); rng = np.random.default_rng(SEED); hist = []
for ep in range(EP):
    p = rng.permutation(TRI)
    for a in range(0, len(p) - BS + 1, BS):
        j = mx.array(p[a:a + BS]); l, g = vg(m, XA[j], LA[j]); opt.update(m, g); mx.eval(m.parameters(), opt.state)
    v, tr = evalkl(XA, LA, VAI), evalkl(XA, LA, TRS); hist.append((tr, v))
    if v < best[0]: best = (v, ep); snap = snapf()
    print(f"  epoch {ep}: train {tr:.4f} val {v:.4f}  {time.time() - t0:.0f}s", flush=True)
m.load_weights(list((k, mx.array(v)) for k, v in snap.items()), strict=False); s = evalkl(XE, LE)
tag = f"{START}_{DROP}_{'-'.join(TRAIN)}_bs{BS}_lr{LR:g}_cos{COS}_sf{SF}_s{SEED}_more{MORE}_feat{FEAT:g}_lora{LORA}"
print(f"[{tag}] best epoch {best[1]} val {best[0]:.4f} -> SEALED KL {s:.4f} (start {start_sealed:.4f}); won back {(start_sealed - s) / start_sealed:.3f}")
np.savez(os.path.join(D, f"tr5b_{tag}.npz"), **snap); json.dump({"tag": tag, "hist": hist, "best": best, "sealed": s, "start": start_sealed, "ntr": int(ntr)}, open(os.path.join(RES, f"vision-d5b-{tag}.json"), "w"))
