"""BEAT I4b microscope (on-set fit): replace the trained k = 3 read by its member-span projection, keep the trained write."""
import numpy as np, torch
from inception import *
from family import Data, Family
from w_fisher import load_build, real_kl
m = load_model(); D = Data(seed=0); fam = list(np.load("cache/s1_fine.npz")["fam"])
W5 = m.mixed3b_5x5_pre_relu_conv.weight.detach()[[i - 320 for i in fam]]; b5 = m.mixed3b_5x5_pre_relu_conv.bias.detach()[[i - 320 for i in fam]]
flat = lambda t: t[:, :, 4:24, 4:24].permute(1, 0, 2, 3).reshape(t.shape[1], -1).cpu().double().numpy()
M = load_build(m, fam, "L-k3-m-lr3e-2"); ix = np.sort(np.random.default_rng(41).permutation(D.tr)[:1500])[:200]
Z, P = [], []
with torch.no_grad():
    for i in range(0, len(ix), 50):
        ins = inputs_3b(m, to_3a(m, prep(D.X[ix[i:i + 50]]))); Z.append(flat(pre_3b(m, ins[0])[:, fam])); P.append(flat(M.pre(ins)))
Z, P = np.concatenate(Z, 1), np.concatenate(P, 1); Dm = np.concatenate([Z.T, np.ones((Z.shape[1], 1))], 1)
Wt = torch.tensor(np.stack([np.linalg.lstsq(Dm[P[j] > 0], P[j][P[j] > 0], rcond=None)[0] for j in range(3)]), dtype=torch.float32, device=DEV)  # (3, 11), ON-set fit
read = torch.einsum("ji,iabc->jabc", Wt[:, :10], W5); bias = Wt[:, :10] @ b5 + Wt[:, 10]
S = Family(m, fam, 3).to(DEV); cols = M.weights()[2]; S.set_weights(read, bias, cols)
zc = lambda fm: [torch.zeros(c.shape[0], device=DEV) for c in fm.weights()[2]]
builds = {"M trained": (M, M.weights()[2], zc(M)), "M span read + trained write": (S, cols, zc(S))}
K = real_kl(m, fam, D, D.dv, builds)
for k, v in K.items(): print(f"{k:30s} dev-{len(v)} KL {v.mean():.4f} (se {v.std() / np.sqrt(len(v)):.4f})")
d = K["M span read + trained write"] - K["M trained"]; print(f"paired span - trained {d.mean():+.4f} (se {d.std() / np.sqrt(len(d)):.4f})")
np.savez("cache/u3b_span_on.npz", W=Wt.cpu().numpy(), **{k.replace(' ', '_'): v for k, v in K.items()})
