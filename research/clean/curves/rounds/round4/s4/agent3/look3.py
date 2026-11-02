# serves: DQ19
import sys
sys.path.insert(0, '/Users/shonusengupta/Cale_Experimentation/research/clean/curves/rounds/round4/s4/agent3')
from common import *
sb, meta, lab = load()
A = splitA()
K, C = sb.shape
off = ~torch.eye(K, C, dtype=torch.bool)
Wt = (sb * off)[A].numpy()
p = Wt / Wt.sum()
print("effective n entries", 1 / (p ** 2).sum(), "of", p.size)
rowp = Wt / Wt.sum(1, keepdims=True)
srt = -np.sort(-rowp, 1)
print("row off-diag mass in top m entries (mean over rows):", {m: round(float(srt[:, :m].sum(1).mean()), 3) for m in [1, 3, 5, 10, 20, 50, 100, 300]})
# share of weighted SS contributed by entries by rank within row, for alpha+beta residual
R = np.load(sys.path[0] + "/R_ab.npy")
ss = Wt * R ** 2
order = np.argsort(-Wt, 1)
ssr = np.take_along_axis(ss, order, 1)
cum = np.cumsum(ssr.sum(0)) / ss.sum()
print("share of SS by within-row weight rank:", {m: round(float(cum[m - 1]), 3) for m in [1, 5, 10, 20, 50, 100, 300, 1000]})
# row diag mass
print("row total mass off-diag mean", Wt.sum(1).mean())
