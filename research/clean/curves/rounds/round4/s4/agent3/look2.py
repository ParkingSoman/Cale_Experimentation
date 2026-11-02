# serves: DQ19
import sys
sys.path.insert(0, '/Users/shonusengupta/Cale_Experimentation/research/clean/curves/rounds/round4/s4/agent3')
from common import *
sb, meta, lab = load()
A = splitA()
K, C = sb.shape
off = ~torch.eye(K, C, dtype=torch.bool)
W = (sb * off)[A].numpy()
R = np.load(sys.path[0] + "/R_ab.npy")
Wr = np.sqrt(W) * R
u, s, vt = np.linalg.svd(Wr, full_matrices=False)
print("s", np.round(s[:8], 1))
# is comp k symmetric: u_k (rows, classes A) vs v_k at columns A
for k in range(4):
    print(k, "corr u vs v at same class", np.corrcoef(u[:, k], vt[k][A])[0, 1])
# what is comp 1? relation to word log-mass, beta, row diag mass
beta = (np.log(sb.numpy()[A]) * W).sum(0) / W.sum(0).clip(1e-300)
print("corr v1 with colmass", np.corrcoef(vt[0], np.log(sb.numpy().mean(0)))[0, 1])
dm = sb.numpy()[A, A]
print("corr u1 with diag mass", np.corrcoef(u[:, 0], dm)[0, 1], "with log diag", np.corrcoef(u[:, 0], np.log(dm))[0, 1])
# predicted: R ~ a_c * v1 ; rows uniform scale?
print("u1 sign mix", (u[:, 0] > 0).mean(), "v1 sign mix", (vt[0] > 0).mean())
# rank-1 improvement of weighted rms
for k in [0, 1, 2, 4, 8, 16]:
    rec = (u[:, :k] * s[:k]) @ vt[:k]
    res = Wr - rec
    print(k, "rms", np.sqrt((res ** 2).sum() / W.sum()))
