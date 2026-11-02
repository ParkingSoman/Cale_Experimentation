"""D3 diag for the block merge: (a) per group: corr(h_G, sum of members' outputs), members' pairwise corr; (b) share of the
block's stream output kept: cut vs merge, raw and with the stream's top direction removed (the essay's unread
'saliency' dimension); (c) the same in the readers' view (next block's LN-scaled fc1 reads, as a linear map)."""
import sys; sys.argv = ["d3_merge.py", "16", "1536", "ward"]
src = open("d3_merge.py").read(); exec(src.split("def lin(")[0])
O = Hh @ W2.T; Oc = O - O.mean(0)
en = np.linalg.norm(Hh, axis=0) * np.linalg.norm(W2, axis=0); keep = np.argsort(-en)[:1536]; drop = np.setdiff1d(np.arange(H), keep)
Ocut = Hh[:, keep] @ W2[:, keep].T; Ocut = Ocut - Ocut.mean(0); Omer = (Hn - Hn.mean(0)) @ nW2.T
U, s_, Vt = np.linalg.svd(Oc[::5], full_matrices=False); top = Vt[0]
def share(A, ruler=None):
    E, R = A - Oc, Oc
    if ruler is not None: E, R = E @ ruler.T, R @ ruler.T
    return 1 - (E ** 2).sum() / (R ** 2).sum()
Pn = np.eye(768) - np.outer(top, top)
nxt = m.blocks[B + 1]; Rd = np.array(nxt.fc1.weight, np.float64) * np.array(nxt.norm.weight, np.float64)[None]
print(f"top stream direction carries {s_[0] ** 2 / (s_ ** 2).sum():.3f} of the block output variance")
for nm, A in (("cut", Ocut), ("merge", Omer)):
    print(f"{nm:6s}: kept share raw {share(A):.3f} | without top direction {share(A, Pn):.3f} | readers' view (block {B + 1} fc1 reads) {share(A, Rd @ Pn):.3f}")
mult = [g for g in groups if len(g) > 1]; cg = [np.corrcoef(Hn[:, i], Hh[:, g].sum(1))[0, 1] for i, g in enumerate(groups) if len(g) > 1]
pc = [np.corrcoef(Hh[:, g].T)[np.triu_indices(len(g), 1)].mean() for g in mult]
print(f"groups with >1 member: {len(mult)}; corr(h_G, sum of members) median {np.median(cg):.3f}; members' pairwise corr median {np.median(pc):.3f}")
print(f"energy of the dropped half (cut) / total: {(en[drop] ** 2).sum() / (en ** 2).sum():.3f}")
