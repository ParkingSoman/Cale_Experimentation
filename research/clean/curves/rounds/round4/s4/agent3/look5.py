# serves: DQ19
import sys
sys.path.insert(0, '/Users/shonusengupta/Cale_Experimentation/research/clean/curves/rounds/round4/s4/agent3')
from common import *
sb, meta, lab = load()
A = splitA()
K, C = sb.shape
off = ~torch.eye(K, C, dtype=torch.bool)
W = (sb * off)[A].numpy(); L = sb.log()[A].numpy()
al = np.zeros(len(A)); be = np.zeros(C)
for _ in range(60):
    al = (W * (L - be[None])).sum(1) / W.sum(1)
    be = (W * (L - al[:, None])).sum(0) / np.maximum(W.sum(0), 1e-300)
R = L - al[:, None] - be[None]
Rb = R[:, A]; Wb = W[:, A]            # A x A block
Rt, Wt_ = Rb.T, Wb.T
m = ~np.eye(len(A), dtype=bool)
w = np.minimum(Wb, Wt_) * m          # weight only pairs where both directions carry mass
def wcorr(x, y, w):
    mx = (w * x).sum() / w.sum(); my = (w * y).sum() / w.sum()
    return (w * (x - mx) * (y - my)).sum() / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum())
print("weighted corr R[c,i] vs R[i,c] (alpha+beta resid), min-weights:", wcorr(Rb, Rt, w))
# head-only: pairs where c->i is in c's top 5
rank = np.argsort(np.argsort(-Wb, 1), 1)
head = (rank < 5) & m
print("pairs in row top5:", head.sum(), " of which reverse also in row top5:", (head & head.T).sum() / head.sum())
head10 = (rank < 10) & m
print("top10 reverse also top10:", (head10 & head10.T).sum() / head10.sum())
# top-1: reverse relation
top1 = rank == 0
print("top1 reciprocal (i's top1 is c):", (top1 & top1.T).sum() / top1.sum())
