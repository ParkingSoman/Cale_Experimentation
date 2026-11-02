"""32.2 (Part VIII): per-question same-image bias contribution phi(x_j) (368 numbers each; raw and factorised via 4
derangement partners) and reader j's own uniform-metric gradient energy E_j on its own question. usage: c57_phi.py N"""
import sys, numpy as np, torch, time
N = int(sys.argv[1])
SRC = open("c41_split.py").read(); exec(SRC.split("ref = torch.load")[0])
vS = Vs["S"] * mask
X = np.load("cache/c45_readerq_368.npy")[:N]
def bcol(T3): return torch.cat([T3[:, :304, 480], T3[:, 304:, 961]], 1)          # (B, 368) bias column per reader
def contrib(Xg, Xy):
    Gm = gam2(Xg, Xy); sc = (Gm * vS).sum((2, 3)); full = torch.einsum("bc,bcjk->bjk", sc, Gm)
    return bcol(full), (Gm ** 2).sum((1, 3))                                     # (B, 368), energy per reader (B, 368)
r = np.random.default_rng(41); DERS = []
while len(DERS) < 4:
    pp = r.permutation(N)
    if (pp != np.arange(N)).all(): DERS.append(pp)
raw, fact, en = [], [], []
for i in range(0, N, 8):
    torch.manual_seed(i); a, e = contrib(X[i:i + 8], X[i:i + 8]); raw.append(a.cpu()); en.append(e.cpu())
    f = sum(contrib(X[i:i + 8], X[der[i:i + 8]])[0] for der in DERS) / 4; fact.append(f.cpu())
raw, fact, en = torch.cat(raw), torch.cat(fact), torch.cat(en)
torch.save(dict(raw=raw, fact=fact, en=en), f"cache/c57_phi_{N}.pt")
print(f"N {N}: raw {tuple(raw.shape)}; |raw| mean {float(raw.norm(dim=1).mean()):.3g}; own-energy E_jj mean {float(en[range(N), range(N)].mean()):.3g}; nan {bool(torch.isnan(raw).any())} ({time.time() - t0:.0f}s)")
print("done")
