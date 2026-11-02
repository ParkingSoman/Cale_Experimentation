"""D2 reads: compare each net's reads with the 16 ideal coarse templates (+/- A_c + mean_f B_cf). Per net: median best
|cos|, how many of the 16 templates are some neuron's best match (coverage), and the bias/threshold scale."""
import os, sys
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d2_lda.py")).read().split("for s in range(3):")[0])
tmpl = np.concatenate([Ac + Bf.mean(1), -(Ac + Bf.mean(1))]); tmpl /= np.linalg.norm(tmpl, axis=1, keepdims=True)
fine = np.concatenate([(Ac[:, None] + Bf).reshape(-1, D), -(Ac[:, None] + Bf).reshape(-1, D)]); fine /= np.linalg.norm(fine, axis=1, keepdims=True)
def look(n, name):
    U = n["W1"] / np.linalg.norm(n["W1"], axis=1, keepdims=True); C = U @ tmpl.T; F = U @ fine.T
    bm = C.argmax(1); print(f"  {name:28s} best coarse-template cos median {np.median(C.max(1)):.2f}; covered {len(set(bm))}/16; best fine-template cos {np.median(F.max(1)):.2f}; second-best coarse cos {np.median(np.sort(C, 1)[:, -2]):.2f}")
W = net("gelu_256_s0"); Nn = net("gelu_16_s0"); Hr = gl(Xr @ W["W1"].T + W["b1"]); HN = gl(Xr @ Nn["W1"].T + Nn["b1"])
zs = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9); Cc = zs(Hr).T @ zs(HN) / len(Hr); parts = [np.argsort(-Cc[:, j])[:10] for j in range(16)]
U = W["W1"] / np.linalg.norm(W["W1"], axis=1, keepdims=True); print("  wide neurons: best fine-template cos median %.2f, best coarse %.2f" % (np.median((U @ fine.T).max(1)), np.median((U @ tmpl.T).max(1))))
look(Nn, "trained narrow"); look(build(W, parts), "oracle groups (consensus)"); look(build(W, groups_by(Hr, 16, "ward")), "ward co-firing groups")
ideal = dict(W1=tmpl * 3, b1=np.zeros(16)); look(ideal, "ideal templates")
for th in (0, 1, 2, 3): print(f"  ideal templates (scale 3, bias -{th}) + Bayes write: fine %.3f coarse %.3f KL %.3f" % score(bayes_write(W, dict(W1=tmpl * 3, b1=-th * np.ones(16))), W))
