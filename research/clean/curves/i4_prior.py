"""I4: (a) logit level LSE on real / shuffled / noise / self-made inputs; (b) conv1 filters vs the real patch spectrum."""
import numpy as np, torch
from scipy.stats import spearmanr
from inception import *
from family import Data, pool3
m = load_model(); D = Data(seed=0); rng = np.random.default_rng(41)
def logits(x):
    a3 = to_3a(m, x); ins = inputs_3b(m, a3); h = F.relu(pre_3b(m, ins[0])); h = mixed(m, "mixed4a", pool3(h))
    for n in ["mixed4b", "mixed4c", "mixed4d", "mixed4e"]: h = mixed(m, n, h)
    h = pool3(h)
    for n in ["mixed5a", "mixed5b"]: h = mixed(m, n, h)
    return m.softmax2_pre_activation_matmul(F.avg_pool2d(h, 7).reshape(-1, 1024))
R = np.asarray(D.X[np.sort(rng.choice(D.tr, 256, replace=False))])
shuf = R.reshape(256, -1, 3).copy()
for i in range(256): shuf[i] = shuf[i][rng.permutation(shuf.shape[1])]
sets = {"real": R, "shuffled real": shuf.reshape(R.shape), "uniform noise": rng.integers(0, 256, R.shape, dtype=np.uint8),
        "self-made (E)": np.load("cache/e1_self_256.npy")}
import sys
with torch.no_grad():
    for k, X in (sets.items() if "b" not in sys.argv else []):
        lg = torch.cat([logits(prep(X[i:i + 32])) for i in range(0, 256, 32)]); lse = torch.logsumexp(lg, 1); p = F.softmax(lg, 1)
        print(f"(a) {k:14s} LSE mean {float(lse.mean()):7.2f} sd {float(lse.std()):5.2f} | max-logit {float(lg.max(1).values.mean()):6.2f} | 1-sum p^2 {float((1 - (p ** 2).sum(1)).mean()):.3f}")
# (b) real 7x7x3 patch covariance (lucid scale), conv1 filter energy along its eigenvectors
x = prep(R[:64]); P = F.unfold(x, 7, stride=4).transpose(1, 2).reshape(-1, 147).cpu().double()
C = torch.cov(P.T); ev, U = torch.linalg.eigh(C); ev, U = ev.flip(0), U.flip(1)
K = m.conv2d0_pre_relu_conv.weight.detach().cpu().double().reshape(64, -1)   # (64, 147) in (c, y, x) order like unfold
e = ((K @ U) ** 2).sum(0)
ok = ev > ev.max() * 1e-6; sl = np.polyfit(np.log(ev[ok].numpy()), np.log(e[ok].numpy()), 1)[0]
print(f"(b) conv1 energy vs patch eigenvalue: log-log slope {sl:+.3f}, Spearman {spearmanr(ev[ok], e[ok])[0]:+.3f} (147 directions)")
print("    energy share of top 10 / top 50 eigen-directions: %.3f / %.3f (uniform would be %.3f / %.3f)" % (e[:10].sum() / e.sum(), e[:50].sum() / e.sum(), 10 / 147, 50 / 147))
# whitening reading: K^T K vs C^(-1/2) and C^(-1): cosine between the matrices
G = K.T @ K
for a in [-1.0, -0.5, 0.0, 0.5]:
    Ca = U @ torch.diag(ev.clamp(min=ev.max() * 1e-6) ** a) @ U.T
    print(f"    cos(K^T K, C^{a:+.1f}) = {float((G * Ca).sum() / G.norm() / Ca.norm()):.3f}")
