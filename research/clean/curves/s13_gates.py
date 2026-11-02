"""Which votes does a trained coarse neuron keep? Split each moved fine kernel into positive votes (W+) and negative
votes (W-). Within-family gates = negative votes on input (channel, position) cells where some moved sibling votes
positive. cos of the trained kernel (zero-padded to 13x13 if needed) with: sum W, sum W+, sum W-, sum of
within-family gates, sum of outside gates, for r = 0..4. Null: shuffled orientation labels.
usage: s13_gates.py <tag>"""
import sys, numpy as np, torch
from inception import *
from family import Family
from derive import PREF
m = load_model(); fam = list(np.load("cache/s1_fine.npz")["fam"])
W = m.mixed3b_5x5_pre_relu_conv.weight.detach().cpu().numpy()[[n - 320 for n in fam]]
ck = torch.load(f"cache/ckpt/{sys.argv[1]}.pt", map_location="cpu"); fm = Family(m, ck["fam"], ck["k"], ks=ck.get("ks", 5)); fm.load_state_dict(ck["state"])
K = fm.weights()[0][0].numpy(); ks = K.shape[-1]; KS = 13
Kp = np.zeros((32, KS, KS)); o = (KS - ks) // 2; Kp[:, o:o + ks, o:o + ks] = K
def moved_k(r, th):
    out = np.zeros((10, 32, KS, KS)); c = KS // 2
    for i, t in enumerate(th):
        dy, dx = int(round(r * np.sin(t))), int(round(-r * np.cos(t)))
        out[i, :, c + dy - 2:c + dy + 3, c + dx - 2:c + dx + 3] = W[i]
    return out
cos = lambda a, b: (a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
def parts(r, th):
    Mk = moved_k(r, th); P = np.maximum(Mk, 0); N = np.minimum(Mk, 0)
    sibpos = P.sum(0)[None] - P                                   # positive votes of the OTHER members per cell
    Nin = np.where(sibpos > 0, N, 0); Nout = N - Nin
    return dict(sum=Mk.sum(0), pos=P.sum(0), neg=N.sum(0), gates_in=Nin.sum(0), gates_out=Nout.sum(0), pos_plus_out=P.sum(0) + Nout.sum(0))
rng = np.random.default_rng(0)
for r in [0, 1, 2, 3, 4]:
    p = parts(r, PREF); nul = [parts(r, rng.permutation(PREF)) for _ in range(30)]
    print(f"r={r}: " + "  ".join(f"{k} {cos(Kp, v):+.3f} (null {np.mean([cos(Kp, n[k]) for n in nul]):+.3f})" for k, v in p.items()))
