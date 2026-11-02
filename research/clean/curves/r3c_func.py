"""R3c (Prop. 11.1, functional): score B3 rulers with the reader change Delta-W projected on span(W_F), and with Delta-W = 0."""
import sys, numpy as np, torch
from inception import *
import b3_train
from family import Data, evaluate, IN4A
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
WF = [getattr(m, n).weight.detach()[:, fam, 0, 0] for n in IN4A]; Q, _ = torch.linalg.qr(torch.cat(WF, 0)); sizes = [w.shape[0] for w in WF]
for tag in sys.argv[1:]:
    fm = b3_train.load_b3(m, tag); orig = [d.detach().clone() for d in fm.dW]
    dW = torch.cat([d * s for d, s in zip(orig, fm.cscale)], 0); P = Q @ (Q.T @ dW)
    res = {"trained": evaluate(m, fm, D, D.dv)}
    with torch.no_grad():
        for d, p, s in zip(fm.dW, torch.split(P, sizes, 0), fm.cscale): d.copy_(p / s)
    res["dW in span(W_F)"] = evaluate(m, fm, D, D.dv)
    with torch.no_grad():
        for d in fm.dW: d.zero_()
    res["dW = 0"] = evaluate(m, fm, D, D.dv)
    for k, v in res.items(): print(f"{tag} {k:18s} dev-5000 KL {v[0]:.4f} (se {v[1]:.4f})", flush=True)
