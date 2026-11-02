"""Phase 2, fine curve family: kernel similarity, co-firing, 4a read columns, bias, firing rate. Ordered by preferred
orientation (from G0 tuning, radius 60)."""
import numpy as np, torch, json
from inception import *
from family import IN4A
m = load_model(); X, y = load_images(); tr, dv, se = split(y)
Z = np.load("results_g0_tuning.npz"); th = Z["th"]; tun = Z["60"]
pref = {n: int(th[tun[:, n].argmax()]) for n in CURVES}
fam = sorted(CURVES, key=lambda n: pref[n]); print("order by preferred orientation:", [(n, pref[n]) for n in fam])
W = m.mixed3b_5x5_pre_relu_conv.weight.detach().cpu()[[n - 320 for n in fam]].flatten(1).numpy()
b = m.mixed3b_5x5_pre_relu_conv.bias.detach().cpu()[[n - 320 for n in fam]].numpy()
Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
np.set_printoptions(precision=2, suppress=True, linewidth=160)
print("kernel norms:", np.linalg.norm(W, axis=1)); print("biases:", b)
print("kernel cosine (rows/cols in orientation order):\n", Wn @ Wn.T)
# mean kernel and how much of each kernel is the shared part
mk = W.mean(0); print("share of each kernel's energy along the family mean kernel:", ((W @ mk / np.linalg.norm(mk))**2 / (W**2).sum(1)))
# natural statistics: activations at all positions on 1000 train images
rng = np.random.default_rng(0); idx = np.sort(rng.choice(tr, 1000, replace=False)); A = []
with torch.no_grad():
    for i in range(0, 1000, 100):
        z = pre_3b(m, to_3a(m, prep(X[idx[i:i+100]])))[:, fam]; A.append(z.permute(0, 2, 3, 1).reshape(-1, 10).cpu().numpy())
A = np.concatenate(A); R = np.maximum(A, 0)
print("firing rate (pre>0):", (A > 0).mean(0)); print("mean pre:", A.mean(0)); print("std pre:", A.std(0))
print("pre-activation correlation:\n", np.corrcoef(A.T)); print("post-ReLU correlation:\n", np.corrcoef(R.T))
on = A > 0; k = on.sum(1)
print("positions with n family members on: ", {i: round(float((k == i).mean()), 4) for i in range(11)})
# 4a reads
for nm in IN4A:
    C = getattr(m, nm).weight.detach().cpu()[:, fam, 0, 0].numpy()   # (out, 10)
    Cn = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)
    print(nm, "column norms:", np.linalg.norm(C, axis=0)); print(" column cosine:\n", Cn.T @ Cn)
np.savez("cache/s1_fine.npz", A=A[:200000], fam=fam, W=W, b=b)
