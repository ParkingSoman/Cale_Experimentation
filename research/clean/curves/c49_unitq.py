"""Questions at a layer (Part VIII, DERIVED rule): one picture per unit of LAYER (N units, rng 49 subset if N < all),
maximise the unit's activity at the centre of its map. Plain pixels, lucent standard transforms, Adam .05, STEPS
steps, feature-vis ReLU (same as e1). usage: c49_unitq.py LAYER N STEPS -> cache/c49_q_LAYER_N[_sSTEPS].npy"""
import sys, time, numpy as np, torch
from lucent.optvis import param, transform
from inception import *
from family import pool3
import lucent.optvis.param.spatial as _sp, lucent.optvis.param.color as _co, lucent.optvis.transform as _tr
for _mod in (_sp, _co, _tr): _mod.device = DEV
LAYER, N, STEPS, BS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), 32
ORDER = ["mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e"]
mr = load_model(redirected=True)
for p in mr.parameters(): p.requires_grad_(False)
def upto(x):
    a3 = to_3a(mr, x); ins = inputs_3b(mr, a3); h = pool3(F.relu(pre_3b(mr, ins[0])))
    for n in ORDER[:ORDER.index(LAYER) + 1]: h = mixed(mr, n, h)
    return h
with torch.no_grad(): C = upto(torch.zeros(1, 3, 224, 224, device=DEV)).shape[1]
units = np.arange(C) if N >= C else np.sort(np.random.default_rng(49).choice(C, N, replace=False))
tf = transform.compose(transform.standard_transforms); out = []; t0 = time.time()
for i in range(0, len(units), BS):
    us = torch.tensor(units[i:i + BS], device=DEV); torch.manual_seed(i)
    params, img_f = param.image(224, batch=len(us), fft=False, decorrelate=False); opt = torch.optim.Adam(params, lr=0.05)
    for s in range(STEPS):
        U = upto(tf(img_f()) * 255.0 - 117.0); c = U.shape[-1] // 2
        loss = -U[:, :, c, c].gather(1, us[:, None]).sum(); opt.zero_grad(); loss.backward(); opt.step()
    out.append((img_f().detach().permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
    print(f"  {i + len(us)}/{len(units)}  {time.time() - t0:.0f} s", flush=True)
np.save(f"cache/c49_q_{LAYER}_{N}{'' if STEPS == 256 else f'_s{STEPS}'}.npy", np.concatenate(out)); np.save(f"cache/c49_q_{LAYER}_{N}_units.npy", units); print("done")
