"""Phase 2: read a trained coarse family (checkpoint) against the fine family, in the shared basis.
usage: s2_coarse.py <ckpt tag>"""
import sys, json, numpy as np, torch
from inception import *
from family import Family, IN4A
from study import tuning, top_spots, sheet, fine_fn

tag = sys.argv[1]
ck = torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu")
m = load_model(); X, y = load_images(); tr, dv, se = split(y)
fm = Family(m, ck["fam"], ck["k"], ks=ck.get("ks", 5)).to(DEV); fm.load_state_dict(ck["state"])
read, bias, cols = [t.cpu().numpy() for t in fm.weights()[:2]] + [[c.cpu().numpy() for c in fm.weights()[2]]]
S1 = np.load("cache/s1_fine.npz"); fam = list(S1["fam"]); W = S1["W"]; b = S1["b"]
np.set_printoptions(precision=3, suppress=True, linewidth=160)
out = {"tag": tag, "fam_order": [int(f) for f in fam]}
k = ck["k"]; R = read.reshape(k, -1)
print("order:", fam)
# (b) read kernel in the span of the ten curve kernels
coef, res, *_ = np.linalg.lstsq(W.T, R.T, rcond=None)          # (10, k)
inspan = 1 - ((R.T - W.T @ coef)**2).sum(0) / (R**2).sum(1)
W96 = m.mixed3b_5x5_pre_relu_conv.weight.detach().cpu().flatten(1).numpy()
c96, *_ = np.linalg.lstsq(W96.T, R.T, rcond=None); in96 = 1 - ((R.T - W96.T @ c96)**2).sum(0) / (R**2).sum(1)
print("coarse kernel norm:", np.linalg.norm(R, axis=1), " bias:", bias)
print("share of coarse kernel energy in span of 10 curve kernels:", inspan, " in span of all 96 5x5 kernels:", in96)
print("coefficients on the curve kernels (orientation order), per coarse neuron:\n", coef.T)
cos = (R / np.linalg.norm(R, axis=1, keepdims=True)) @ (W / np.linalg.norm(W, axis=1, keepdims=True)).T
print("cosine with each curve kernel:\n", cos)
out.update(inspan=inspan.tolist(), in96=in96.tolist(), coef=coef.T.tolist(), cos=cos.tolist(), bias=bias.tolist())
# (f) 4a columns vs original columns
for nm, C in zip(IN4A, cols):
    C0 = getattr(m, nm).weight.detach().cpu()[:, fam, 0, 0].numpy()
    beta, *_ = np.linalg.lstsq(C0, C, rcond=None); ex = 1 - ((C - C0 @ beta)**2).sum(0) / (C**2).sum(0)
    print(f"{nm}: coarse column norm {np.linalg.norm(C, axis=0)}, explained by original 10 columns {ex}, beta:\n", beta.T)
    out[nm] = dict(beta=beta.T.tolist(), explained=ex.tolist())
# (a) tuning
tf = tuning(m, fine_fn(m, fam)); tc = tuning(m, lambda ins: fm.pre(ins))
for r in tf:
    print(f"radius {r}: coarse response by orientation (0..350 step 10):", tc[r][:, :].T.round(0))
    print(f"   fine family max over members:", tf[r].max(1).round(0), " sum:", tf[r].sum(1).round(0))
out["tuning_coarse"] = {str(r): tc[r].tolist() for r in tc}; out["tuning_fine"] = {str(r): tf[r].tolist() for r in tf}
# (d) natural statistics
rng = np.random.default_rng(0); idx = np.sort(rng.choice(tr, 600, replace=False)); A, Cc = [], []
with torch.no_grad():
    for i in range(0, len(idx), 100):
        ins = inputs_3b(m, to_3a(m, prep(X[idx[i:i+100]])))
        A.append(pre_3b(m, ins[0])[:, fam].permute(0, 2, 3, 1).reshape(-1, 10).cpu().numpy())
        Cc.append(fm.pre(ins).permute(0, 2, 3, 1).reshape(-1, k).cpu().numpy())
A = np.concatenate(A); Cc = np.concatenate(Cc); Ar = np.maximum(A, 0); Cr = np.maximum(Cc, 0)
print("coarse firing rate:", (Cc > 0).mean(0), " fine mean rate:", (A > 0).mean())
cc = np.array([[np.corrcoef(Cr[:, j], Ar[:, i])[0, 1] for i in range(10)] for j in range(k)])
print("post-ReLU corr coarse vs each fine:\n", cc)
print("corr coarse vs family sum of post-ReLU:", [np.corrcoef(Cr[:, j], Ar.sum(1))[0, 1] for j in range(k)],
      " vs family max:", [np.corrcoef(Cr[:, j], Ar.max(1))[0, 1] for j in range(k)])
nOn = (A > 0).sum(1)
print("coarse rate given n fine on:", [[round(float((Cc[nOn == n, j] > 0).mean()), 3) for n in range(5)] for j in range(k)])
out.update(rate=(Cc > 0).mean(0).tolist(), corr_fine=cc.tolist())
json.dump(out, open(f"../../../results/curves-s2-{tag}.json", "w"), indent=1)
# (g) crops
sp = top_spots(m, lambda ins: fm.pre(ins), X, np.sort(rng.choice(tr, 3000, replace=False)), n_keep=10)
print("sheet:", sheet(X, sp, f"cache/sheet-{tag}.png"))
