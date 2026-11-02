"""Statement 16.8: factorised formula with the WEIGHTS metric (no image in the metric). Metric over the family's 10
signals x 196 positions = sum over 4a branch units of effective-kernel products (all four branches), placed on the
x-signal block; activity moments from n images as in x5f_factor.py."""
import numpy as np, torch
from inception import *
m0 = load_model()
fam = list(np.load("cache/s1_fine.npz")["fam"])
w = lambda n: getattr(m0, n).weight.detach().cpu().double()
WF = lambda n: w(n)[:, fam, 0, 0]
ker = [(torch.einsum("ubyx,bf->ufyx", w("mixed4a_3x3_pre_relu_conv"), WF("mixed4a_3x3_bottleneck_pre_relu_conv")), 1),
       (torch.einsum("ubyx,bf->ufyx", w("mixed4a_5x5_pre_relu_conv"), WF("mixed4a_5x5_bottleneck_pre_relu_conv")), 2),
       (WF("mixed4a_1x1_pre_relu_conv")[:, :, None, None], 0),
       ((WF("mixed4a_pool_reduce_pre_relu_conv") / 9)[:, :, None, None].expand(-1, -1, 3, 3), 1)]
Mw = torch.zeros(10, 14, 14, 10, 14, 14, dtype=torch.float64)
for k, h in ker:                        # unit at centre c reads family f at c + delta with weight k[:, f, delta]
    s = 2 * h + 1
    for cy in range(14):
        for cx in range(14):
            pos = [(cy + a - h, cx + b - h, a, b) for a in range(s) for b in range(s) if 0 <= cy + a - h < 14 and 0 <= cx + b - h < 14]
            V = torch.zeros(k.shape[0], 10, 14, 14, dtype=torch.float64)
            for y, x, a, b in pos: V[:, :, y, x] = k[:, :, a, b]
            V = V.reshape(k.shape[0], 1960); Mw += (V.T @ V).reshape(10, 14, 14, 10, 14, 14)
Kb = torch.zeros(2, 10, 196, 2, 10, 196); Kb[0, :, :, 0, :, :] = Mw.reshape(10, 196, 10, 196).float()
torch.save(Kb.reshape(3920, 3920), "cache/x5h_kweights.pt")
src = open("x5f_factor.py").read()
src = src.replace('Kb = torch.load("cache/x5_kbar.pt").reshape(2, 10, 196, 2, 10, 196).to(DEV)', 'Kb = torch.load("cache/x5h_kweights.pt").reshape(2, 10, 196, 2, 10, 196).to(DEV)')
src = src.replace('f"cache/x5f_{k}.pt"', 'f"cache/x5h_{k}.pt"').replace('"cache/x5f_factor.npz"', '"cache/x5h_wbuild.npz"')
src = src.replace('"factor_n2000"', '"wmetric_n2000"').replace('"factor_n256"', '"wmetric_n256"')
src = src.replace('ref = {"level0p"', 'ref = {"factor_kbar": np.load("cache/x5f_factor.npz")["factor_n2000"], "level0p"')
exec(src)
