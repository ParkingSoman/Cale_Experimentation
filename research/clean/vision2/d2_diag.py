"""D2 diag: which step of the build fails? (a) oracle groups = the trained narrow neurons' own top-10 wide parts;
(b) per built neuron: corr of h_G with its group's activation sum; (c) share of the group's output (W2 H) captured by the
rank-1 write; (d) overlap of the trained neurons' part lists (are narrow neurons a PARTITION of the wide net?)."""
import os, sys
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d2_build.py")).read().split('if __name__ == "__main__":')[0])
W = net("gelu_256_s0"); Nn = net("gelu_16_s0"); Z = Xr @ W["W1"].T + W["b1"]; H = gl(Z); HN = gl(Xr @ Nn["W1"].T + Nn["b1"])
zs = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9); C = zs(H).T @ zs(HN) / len(H)
for K in (5, 10, 20, 40):
    parts = [np.argsort(-C[:, j])[:K] for j in range(16)]; cover = len(set(np.concatenate(parts)))
    n = build(W, parts); print(f"oracle groups top-{K}: covers {cover} wide neurons; built: fine %.3f coarse %.3f KL %.3f" % score(n, W))
parts = [np.argsort(-C[:, j])[:10] for j in range(16)]; n = build(W, parts); HG = gl(Xr @ n["W1"].T + n["b1"])
print("built vs group sum corr (median): %.3f; built vs trained narrow neuron corr (median): %.3f" % (
    np.median([np.corrcoef(HG[:, g], H[:, p].sum(1))[0, 1] for g, p in enumerate(parts)]), np.median([np.corrcoef(HG[:, g], HN[:, g])[0, 1] for g in range(16)])))
cap = []
for g, p in enumerate(parts):
    O = (H[:, p] - H[:, p].mean(0)) @ W["W2"][:, p].T; Oh = np.outer(HG[:, g] - HG[:, g].mean(), n["W2"][:, g]); cap.append(1 - ((O - Oh) ** 2).sum() / (O ** 2).sum())
print("share of each group's output captured by its rank-1 write (median): %.3f" % np.median(cap))
# what does the TRAINED narrow net's write look like vs the wide parts' writes?
for g in range(3):
    p = parts[g]; wt = Nn["W2"][:, g]; wb = n["W2"][:, g]; print(f"  neuron {g}: cos(trained write, built write) {wt @ wb / np.linalg.norm(wt) / np.linalg.norm(wb):.2f}")
