"""E1: inputs the network makes for itself (DERIVED: weights + a fixed, published generation rule; no image set).
Rule (stated once, never tuned): for class y (classes spread evenly over 1000), maximize the logit of y on a plain
pixel image (lucent param.image fft=False, decorrelate=False: no 1/f law, no ImageNet colour matrix), lucent's
standard transforms (pad/jitter/scale/rotate), Adam lr 0.05, STEPS steps, lucid's feature-vis ReLU.
usage: e1_selfmade.py N STEPS  -> cache/e1_self_N.npy (uint8 NHWC) + labels"""
import sys, time, numpy as np, torch
from lucent.optvis import param, transform
from inception import *
import lucent.optvis.param.spatial as _sp, lucent.optvis.param.color as _co, lucent.optvis.transform as _tr
for _mod in (_sp, _co, _tr):
    _mod.device = DEV
N, STEPS, BS = int(sys.argv[1]), int(sys.argv[2]), 32
m = load_model(redirected=True)
def full(x):   # intact logits from a lucid-scale image
    a3 = to_3a(m, x); ins = inputs_3b(m, a3); h = F.relu(pre_3b(m, ins[0]))
    from family import pool3
    x4 = pool3(h); h = mixed(m, "mixed4a", x4)
    for n in ["mixed4b", "mixed4c", "mixed4d", "mixed4e"]: h = mixed(m, n, h)
    h = pool3(h)
    for n in ["mixed5a", "mixed5b"]: h = mixed(m, n, h)
    return m.softmax2_pre_activation_matmul(F.avg_pool2d(h, 7).reshape(-1, 1024))
ys = np.linspace(0, 999, N).round().astype(int); out = np.zeros((N, 224, 224, 3), np.uint8); t0 = time.time()
tf = transform.compose(transform.standard_transforms)
for i in range(0, N, BS):
    torch.manual_seed(i); y = torch.tensor(ys[i:i + BS], device=DEV); b = len(y)
    params, img_f = param.image(224, batch=b, fft=False, decorrelate=False)
    opt = torch.optim.Adam(params, lr=0.05)
    for s in range(STEPS):
        lg = full(tf(img_f()) * 255.0 - 117.0)
        loss = -lg.gather(1, y[:, None]).sum(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        im = img_f(); acc = float((full(im * 255.0 - 117.0).argmax(1) == y).float().mean())
    out[i:i + b] = (im.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    print(f"  {i + b}/{N}  {time.time() - t0:.0f} s  top-1 on own class {acc:.2f}", flush=True)
np.save(f"cache/e1_self_{N}.npy", out); np.save(f"cache/e1_self_{N}_y.npy", ys)
