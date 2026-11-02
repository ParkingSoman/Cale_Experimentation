"""D2 Fisher write: the write that minimizes the second-order (Taylor) expansion of KL(wide || new) around the wide
model's own output: sum_n (L_n - W h_n - b)^T F_n (L_n - W h_n - b), F_n = diag(p_n) - p_n p_n^T (wide's probabilities).
One linear solve (vec trick), no knobs, no iteration. Readings on train inputs."""
import os, sys
import numpy as np
from scipy.special import softmax
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d2_build.py")).read().split('if __name__ == "__main__":')[0])
def fisher_write(W, n, N=6000):
    X_ = Xr[:N]; Hn = gl(X_ @ n["W1"].T + n["b1"]); L = logits(W, X_); p = softmax(L, 1); A = np.c_[Hn, np.ones(N)]; k = A.shape[1]; C = L.shape[1]
    # minimize sum_n r_n^T F_n r_n, r_n = L_n - M a_n, M (C x k). F_n = diag(p) - p p^T. Normal equations in vec(M).
    G = np.einsum("ni,nj,nc->icjc", A, A, p).reshape(k * C, k * C) if False else None
    Pa = p[:, :, None] * A[:, None, :]                     # (N, C, k): p_c a_i
    lhs = np.einsum("nci,ncj->cij", Pa, A[:, None, :].repeat(C, 1))   # diag part: sum_n p_c a_i a_j  -> (C, k, k)
    big = np.zeros((C * k, C * k))
    for c in range(C): big[c * k:(c + 1) * k, c * k:(c + 1) * k] = lhs[c]
    Q = Pa.reshape(N, C * k); big -= Q.T @ Q                # minus sum_n (p_c a_i)(p_d a_j)
    FL = p * L - p * (p * L).sum(1, keepdims=True)          # F_n L_n
    rhs = np.einsum("nc,ni->ci", FL, A).reshape(-1)
    # F_n has a null direction (all-ones in class space): fix the gauge with a tiny ridge
    M = np.linalg.solve(big + 1e-6 * np.trace(big) / big.shape[0] * np.eye(C * k), rhs).reshape(C, k)
    return dict(n, W2=M[:, :-1], b2=M[:, -1])
for s in range(3):
    W = net(f"gelu_256_s{s}"); Nn = net(f"gelu_16_s{s}"); Hr = gl(Xr @ W["W1"].T + W["b1"]); HN = gl(Xr @ Nn["W1"].T + Nn["b1"])
    zs = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9); Cc = zs(Hr).T @ zs(HN) / len(Hr)
    print(f"seed {s}: trained 16: fine %.3f coarse %.3f KL %.3f" % score(Nn, W), "| trained reads + Fisher write: fine %.3f coarse %.3f KL %.3f" % score(fisher_write(W, Nn), W))
    parts = [np.argsort(-Cc[:, j])[:10] for j in range(16)]; print("   oracle top-10 groups + Fisher write: fine %.3f coarse %.3f KL %.3f" % score(fisher_write(W, build(W, parts)), W))
    for how in ("average", "ward", "random"):
        print(f"   {how:8s} co-firing groups + Fisher write: fine %.3f coarse %.3f KL %.3f" % score(fisher_write(W, build(W, groups_by(Hr, 16, how))), W))
