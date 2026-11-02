"""Fine family over the orientation circle: pre-activation tuning (centre position 14,14 and best of centre 5x5),
Fourier content, and the family's summed response (is the family's total flat over orientation?)."""
import numpy as np, torch
from inception import *
from stimuli import arc
m = load_model(); S1 = np.load("cache/s1_fine.npz"); fam = list(S1["fam"])
th = np.arange(0, 360, 5); np.set_printoptions(precision=2, suppress=True, linewidth=170)
for r in [40, 60]:
    with torch.no_grad():
        Z = pre_3b(m, to_3a(m, prep(np.stack([arc(t, radius=r) for t in th]))))[:, fam]
    zc = Z[:, :, 14, 14].cpu().numpy()                                  # exact centre
    zr = torch.relu(Z)[:, :, 12:17, 12:17].amax((2, 3)).cpu().numpy()   # best spot near centre
    Fz = np.abs(np.fft.rfft(zc - zc.mean(0), axis=0))[:6]
    print(f"r={r}  share of tuning-curve power (centre pre) in harmonics 1..5 per neuron:\n", (Fz[1:]**2 / (Fz[1:]**2).sum(0)).T)
    tot = zr.sum(1); print(f"r={r} family total (post-ReLU, best spot) over orientation: min/mean/max", tot.min().round(), tot.mean().round(), tot.max().round(), " cv", (tot.std()/tot.mean()).round(3))
    print(f"r={r} family max over members: cv", (zr.max(1).std()/zr.max(1).mean()).round(3), " n members > 25% of own peak:", np.bincount((zr > 0.25*zr.max(0)).sum(1), minlength=6))
    pk = th[zr.argmax(0)]; print(f"r={r} peaks:", pk, " gaps:", np.diff(np.sort(pk), append=np.sort(pk)[0]+360))
    w = [(zr[:, i] > 0.5*zr[:, i].max()).sum()*5 for i in range(10)]; print(f"r={r} half-max width (deg):", w)
