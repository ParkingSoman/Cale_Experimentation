# serves: DQ19
import sys
sys.path.insert(0, '/Users/shonusengupta/Cale_Experimentation/research/clean/curves/rounds/round4/s4/agent3')
from common import *
sb, meta, lab = load()
A = splitA()
K, C = sb.shape
off = ~torch.eye(K, C, dtype=torch.bool)
W = (sb * off)[A].numpy()
L = sb.log()[A].numpy()
al = np.zeros(len(A)); be = np.zeros(C)
for _ in range(60):
    al = (W * (L - be[None])).sum(1) / W.sum(1)
    be = (W * (L - al[:, None])).sum(0) / np.maximum(W.sum(0), 1e-300)
R = L - al[:, None] - be[None]
print("ab rms", np.sqrt((W * R ** 2).sum() / W.sum()))
print("diag mass mean", float(sb[torch.tensor(A), torch.tensor(A)].mean()))
Wr = np.sqrt(W) * R
u, s, vt = np.linalg.svd(Wr, full_matrices=False)
tot = (s ** 2).sum()
print("spectrum frac", np.round(np.cumsum(s ** 2)[[0, 1, 2, 4, 9, 14, 29, 49, 99, 199]] / tot, 3))
np.save(sys.path[0] + "/R_ab.npy", R)
