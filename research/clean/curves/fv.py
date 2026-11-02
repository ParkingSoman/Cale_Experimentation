"""Feature visualization (Lucent's FFT + colour-decorrelated image, standard jitter/scale/rotate transforms) for any
neuron set ins -> (B, n, 28, 28): maximize the centre position (14, 14) of neuron j.
usage: fv.py fine 379,406   |   fv.py <ckpt tag> 0"""
import sys, numpy as np, torch
from PIL import Image
from lucent.optvis import param, transform
from inception import *
from family import Family
import lucent.optvis.param.spatial as _sp, lucent.optvis.param.color as _co, lucent.optvis.transform as _tr
for _mod in (_sp, _co, _tr):
    _mod.device = DEV
_CC = torch.tensor(_co.color_correlation_normalized.T).to(DEV)
_co._linear_decorrelate_color = lambda t: torch.matmul(t.permute(0, 2, 3, 1), _CC).permute(0, 3, 1, 2)

def visualize(m, fn, j, steps=384, size=128, seed=0):
    torch.manual_seed(seed)
    params, img_f = param.image(size, fft=True, decorrelate=True)
    tf = transform.compose(transform.standard_transforms)
    opt = torch.optim.Adam(params, lr=0.05)
    for _ in range(steps):
        x = tf(img_f()) * 255.0 - 117.0
        z = fn(inputs_3b(m, to_3a(m, x)))
        c = z.shape[-1] // 2
        loss = -z[0, j, c, c]
        opt.zero_grad(); loss.backward(); opt.step()
    return (img_f()[0].permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

if __name__ == "__main__":
    m = load_model(redirected=True)
    what, js = sys.argv[1], [int(t) for t in sys.argv[2].split(",")]
    if what == "fine":
        fn = lambda ins: pre_3b(m, ins[0]); name = "fine"
    else:
        ck = torch.load(f"cache/ckpt/{what}.pt", map_location="cpu", weights_only=False)
        fm = Family(m, ck["fam"], ck["k"], ks=ck.get("ks", 5)).to(DEV); fm.load_state_dict(ck["state"]); fn = fm.pre; name = what
    ims = [visualize(m, fn, j) for j in js]
    path = f"cache/fv-{name}-{'_'.join(map(str, js))}.png"
    Image.fromarray(np.concatenate(ims, 1)).save(path); print(path)
