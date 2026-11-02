"""F1, F2: per-neuron firing frequency and co-firing from the network's own class inputs vs real images."""
import numpy as np, torch
from scipy.stats import spearmanr
from inception import *
from family import Data, pool3
m = load_model(); D = Data(seed=0); S = np.load("cache/e1_self_256.npy", mmap_mode="r")
R = np.asarray(D.X[np.sort(np.random.default_rng(51).choice(D.tr, 256, replace=False))])
def acts(x):
    a3 = to_3a(m, x); h3b = F.relu(pre_3b(m, inputs_3b(m, a3)[0])); h = mixed(m, "mixed4a", pool3(h3b)); out = {"3a": a3, "3b": h3b, "4a": h}
    for n in ["mixed4b", "mixed4c", "mixed4d"]: h = mixed(m, n, h)
    out["4d"] = h; h = mixed(m, "mixed4e", h); h = pool3(h)
    for n in ["mixed5a", "mixed5b"]: h = mixed(m, n, h)
    out["5b"] = h; return out
def stats(X):
    on, mu, n, cof = {}, {}, {}, []
    with torch.no_grad():
        for i in range(0, 256, 32):
            A = acts(prep(X[i:i + 32]))
            for k, a in A.items():
                on[k] = on.get(k, 0) + (a > 0).float().sum((0, 2, 3)).cpu().numpy(); mu[k] = mu.get(k, 0) + a.sum((0, 2, 3)).cpu().numpy()
                n[k] = n.get(k, 0) + a.shape[0] * a.shape[2] * a.shape[3]
            cof.append(A["3b"].permute(1, 0, 2, 3).reshape(480, -1)[:, ::4].cpu().numpy())
    return {k: on[k] / n[k] for k in on}, {k: mu[k] / n[k] for k in mu}, np.corrcoef(np.concatenate(cof, 1))
on_s, mu_s, C_s = stats(S); on_r, mu_r, C_r = stats(R)
print("F1  layer | neurons | Spearman on-rate | Spearman mean act | on-rate ratio self/real: median (IQR) | share within x2")
for k in ["3a", "3b", "4a", "4d", "5b"]:
    ok = (on_r[k] > 1e-4); r = on_s[k][ok] / on_r[k][ok]
    print(f"    {k:4s} | {len(on_r[k]):4d} | {spearmanr(on_s[k], on_r[k])[0]:+.3f} | {spearmanr(mu_s[k], mu_r[k])[0]:+.3f} | "
          f"{np.median(r):.2f} ({np.percentile(r, 25):.2f}-{np.percentile(r, 75):.2f}) | {np.mean((r > .5) & (r < 2)):.2f}")
iu = np.triu_indices(480, 1); a, b = np.nan_to_num(C_s[iu]), np.nan_to_num(C_r[iu])
print(f"F2  3b co-firing (480 x 480, {len(a)} pairs): Pearson {np.corrcoef(a, b)[0, 1]:+.3f}, Spearman {spearmanr(a, b)[0]:+.3f}")
# attenuation ceiling: split-half reliability (128 / 128) of the per-neuron on-rate, on each side
def half(X, lo, hi):
    on, n = {}, {}
    with torch.no_grad():
        for i in range(lo, hi, 32):
            for k, a in acts(prep(X[i:min(i + 32, hi)])).items():
                on[k] = on.get(k, 0) + (a > 0).float().sum((0, 2, 3)).cpu().numpy(); n[k] = n.get(k, 0) + a.shape[0] * a.shape[2] * a.shape[3]
    return {k: on[k] / n[k] for k in on}
s1, s2, r1, r2 = half(S, 0, 128), half(S, 128, 256), half(R, 0, 128), half(R, 128, 256)
print("ceiling layer | split-half Spearman self | real | ceiling sqrt(rel_s rel_r) (Spearman-Brown to 256) | observed | observed / ceiling")
for k in ["3a", "3b", "4a", "4d", "5b"]:
    rs, rr = spearmanr(s1[k], s2[k])[0], spearmanr(r1[k], r2[k])[0]; sb = lambda r: 2 * r / (1 + r)
    ceil = np.sqrt(max(sb(rs), 0) * max(sb(rr), 0)); obs = spearmanr(on_s[k], on_r[k])[0]
    print(f"        {k:4s} | {rs:+.3f} | {rr:+.3f} | {ceil:.3f} | {obs:+.3f} | {obs / ceil:.2f}")
