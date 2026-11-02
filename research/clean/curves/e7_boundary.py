"""E7: boundary inputs made by the network: cross-entropy to 1/2 e_a + 1/2 e_b, b = a's nearest class by last-layer
weights. Plain pixels, lucent standard transforms, Adam lr .05. usage: e7_boundary.py N STEPS [offset]"""
import sys, time, numpy as np, torch
from lucent.optvis import param, transform
from inception import *
from family import pool3
import lucent.optvis.param.spatial as _sp, lucent.optvis.param.color as _co, lucent.optvis.transform as _tr
for _mod in (_sp, _co, _tr): _mod.device = DEV
N, STEPS, BS = int(sys.argv[1]), int(sys.argv[2]), 32; OFF = int(sys.argv[3]) if len(sys.argv) > 3 else 0
m = load_model(redirected=True); lab = np.load("label_map.npy")
W = m.softmax2_pre_activation_matmul.weight.detach()[torch.tensor(lab)]; Wn = W / W.norm(dim=1, keepdim=True); S = Wn @ Wn.T; S.fill_diagonal_(-2)
cls = (np.linspace(0, 999, N, endpoint=False).round().astype(int) + OFF) % 1000; partner = S.argmax(1).cpu().numpy()[cls]
ya, yb = torch.tensor(lab[cls], device=DEV), torch.tensor(lab[partner], device=DEV)
def full(x):
    a3 = to_3a(m, x); h = F.relu(pre_3b(m, inputs_3b(m, a3)[0])); h = mixed(m, "mixed4a", pool3(h))
    for n in ["mixed4b", "mixed4c", "mixed4d", "mixed4e"]: h = mixed(m, n, h)
    h = pool3(h)
    for n in ["mixed5a", "mixed5b"]: h = mixed(m, n, h)
    return m.softmax2_pre_activation_matmul(F.avg_pool2d(h, 7).reshape(-1, 1024))
out = np.zeros((N, 224, 224, 3), np.uint8); tf = transform.compose(transform.standard_transforms); t0 = time.time()
for i in range(0, N, BS):
    torch.manual_seed(1000 + i + OFF); a, b = ya[i:i + BS], yb[i:i + BS]
    params, img_f = param.image(224, batch=len(a), fft=False, decorrelate=False); opt = torch.optim.Adam(params, lr=0.05)
    for s in range(STEPS):
        lp = F.log_softmax(full(tf(img_f()) * 255.0 - 117.0), 1)
        loss = -0.5 * (lp.gather(1, a[:, None]) + lp.gather(1, b[:, None])).sum(); opt.zero_grad(); loss.backward(); opt.step()
    out[i:i + len(a)] = (img_f().detach().permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    print(f"  {i + len(a)}/{N}  {time.time() - t0:.0f} s", flush=True)
np.save(f"cache/e7_bnd_{N}_{STEPS}_{OFF}.npy", out); np.save(f"cache/e7_bnd_{N}_{STEPS}_{OFF}_pairs.npy", np.stack([cls, partner]))
