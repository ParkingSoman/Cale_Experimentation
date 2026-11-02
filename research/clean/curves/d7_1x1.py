"""Why does the 1x1 write fail? Per 4a 1x1 neuron j: trained column, vmr column, h0 and h1 weights over the curve family,
rho (how well the coarse channel predicts its old family input), its own bias, and its family-input share."""
import sys, numpy as np, torch
from inception import *
from family import Family, Data, IN4A
from d4_write import stats, writes
m = load_model(); D = Data(seed=0, ndev=200); fam = list(np.load("cache/s1_fine.npz")["fam"])
PREF = np.deg2rad(np.array([10, 50, 70, 120, 170, 180, 250, 290, 320, 350]))
ck = torch.load(f"cache/ckpt/{sys.argv[1]}.pt", map_location="cpu", weights_only=False); fm = Family(m, fam, 1, ks=ck.get("ks", 5)).to(DEV); fm.load_state_dict(ck["state"])
st = stats(m, fm, fam, D); res, g = writes(m, fam, st); mu, C = st[0]
A = m.mixed4a_1x1_pre_relu_conv.weight.detach().cpu()[:, fam, 0, 0].double().numpy()
tcol = fm.weights()[2][0][:, 0].cpu().numpy(); vcol = res["vmr"][0][0][:, 0].numpy(); bcol = res["bayes"][0][0][:, 0].numpy()
h0 = A.mean(1); h1 = np.abs(A @ np.exp(1j * PREF)) * 2 / 10
covIc = A @ C[:10, 10]; varI = np.einsum("oi,ij,oj->o", A, C[:10, :10], A); rho = covIc / np.sqrt(varI * C[10, 10])
b4 = m.mixed4a_1x1_pre_relu_conv.bias.detach().cpu().numpy()
sc = np.sqrt(C[10, 10])   # coarse pooled std: columns x sc = reader input std units
np.set_printoptions(precision=2, suppress=True, linewidth=170)
print("corr over 192 neurons: trained col vs vmr col %.2f, vs bayes %.2f, vs h0 %.2f, vs h1 %.2f" % tuple(np.corrcoef(tcol, x)[0, 1] for x in [vcol, bcol, h0, h1]))
d = tcol - vcol; o = np.argsort(-np.abs(d * sc))[:12]
print(" j   trained*sc  vmr*sc  bayes*sc   h0     h1     rho   sd(I_j)  bias")
for j in o: print(f"{j:3d} {tcol[j]*sc:9.1f} {vcol[j]*sc:8.1f} {bcol[j]*sc:8.1f} {h0[j]:6.3f} {h1[j]:6.3f} {rho[j]:6.2f} {np.sqrt(varI[j]):8.1f} {b4[j]:6.1f}")
neg = (tcol * sc < -np.sqrt(varI)); print(f"neurons where trained column is strongly negative (< -sd I_j): {neg.sum()} ; their mean h0 {h0[neg].mean():.3f} h1 {h1[neg].mean():.3f} rho {rho[neg].mean():.2f}")
print("sign agreement trained vs vmr:", (np.sign(tcol) == np.sign(vcol)).mean(), " among |rho|>.3:", (np.sign(tcol) == np.sign(vcol))[np.abs(rho) > .3].mean(), " among |rho|<.1:", (np.sign(tcol) == np.sign(vcol))[np.abs(rho) < .1].mean())
print("ratio trained/bayes (median) among |rho|>.3:", np.median((tcol / bcol)[np.abs(rho) > .3]))
