"""K2 question-maker (Part VIII, DERIVED rule): one picture per family reader j (368): maximise reader j's pre-activation
at the centre of its 14 x 14 map. Plain pixels, lucent standard transforms, Adam .05, STEPS steps, feature-vis ReLU.
usage: c45_readerq.py STEPS -> cache/c45_readerq_368[_sSTEPS].npy"""
import sys, time, numpy as np, torch
from lucent.optvis import param, transform
from inception import *
from w_fisher import intact
import lucent.optvis.param.spatial as _sp, lucent.optvis.param.color as _co, lucent.optvis.transform as _tr
for _mod in (_sp, _co, _tr): _mod.device = DEV
STEPS = int(sys.argv[1]); BS = 32
mr = load_model(redirected=True); fam = list(np.load("cache/s1_fine.npz")["fam"])
for p in mr.parameters(): p.requires_grad_(False)
tf = transform.compose(transform.standard_transforms); out = []; t0 = time.time()
for i in range(0, 368, BS):
    rs = torch.arange(i, min(i + BS, 368), device=DEV); torch.manual_seed(i)
    params, img_f = param.image(224, batch=len(rs), fft=False, decorrelate=False); opt = torch.optim.Adam(params, lr=0.05)
    for s in range(STEPS):
        outs, _, _ = intact(mr, fam, inputs_3b(mr, to_3a(mr, tf(img_f()) * 255.0 - 117.0))); U = torch.cat(outs, 1)
        c = U.shape[-1] // 2; loss = -U[:, :, c, c].gather(1, rs[:, None]).sum(); opt.zero_grad(); loss.backward(); opt.step()
    out.append((img_f().detach().permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
    print(f"  {i + len(rs)}/368  {time.time() - t0:.0f} s", flush=True)
np.save(f"cache/c45_readerq_368{'' if STEPS == 256 else f'_s{STEPS}'}.npy", np.concatenate(out)); print("done")
