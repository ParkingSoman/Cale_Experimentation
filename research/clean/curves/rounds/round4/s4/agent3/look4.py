# serves: DQ19
import sys
sys.path.insert(0, '/Users/shonusengupta/Cale_Experimentation/research/clean/curves/rounds/round4/s4/agent3')
from common import *
import torchvision
names = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
sb, meta, lab = load()
A = splitA()
K, C = sb.shape
S = sb.numpy()
Wc = meta["W"].astype(np.float64)
Wn = Wc / np.linalg.norm(Wc, axis=1, keepdims=True)
cosW = Wn @ Wn.T
rng = np.random.default_rng(1)
for c in rng.choice(A, 12, replace=False):
    row = S[c].copy(); row[c] = 0
    top = np.argsort(-row)[:8]
    print(f"\n[{c}] {names[c]}  diag {S[c,c]:.2f}  top8 mass {row[top].sum()/row.sum():.2f}")
    print("  ", "; ".join(f"{names[i]}({row[i]/row.sum():.2f}, cosW {cosW[c,i]:.2f})" for i in top))
    # are the neighbours mutually related? mean pairwise cosW among top8 vs to c
    sub = cosW[np.ix_(top, top)]
    iu = np.triu_indices(8, 1)
    print("   mean cosW among nbrs", round(float(sub[iu].mean()), 2), " to c", round(float(cosW[c, top].mean()), 2))
