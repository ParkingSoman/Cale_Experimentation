"""D2 Bayes write: if the coarse activations h are Gaussian per class with one shared covariance S, the Bayes posterior
over classes is softmax(W h + b) with W = M S^-1, b_y = -1/2 m_y^T S^-1 m_y + log pi_y (LDA). Class means m_y, priors pi_y
and S are READINGS, weighted by the WIDE model's own probabilities (no true labels, no iteration, no knobs)."""
import os, sys
import numpy as np
from scipy.special import softmax
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d2_build.py")).read().split('if __name__ == "__main__":')[0])
def bayes_write(W, n, soft=True):
    Hn = gl(Xr @ n["W1"].T + n["b1"]); P = softmax(logits(W, Xr), 1) if soft else np.eye(NC * NF)[ytr[:len(Xr)]]
    pi = P.mean(0); M = (P.T @ Hn) / (P.sum(0)[:, None] + 1e-12)            # class means (C, k)
    R = Hn[:, None, :] - M[None]; S = np.einsum("nc,nci,ncj->ij", P, R, R) / len(Hn)   # pooled within-class covariance
    Si = np.linalg.inv(S + 1e-9 * np.trace(S) * np.eye(len(S))); Wt = M @ Si; b = -0.5 * np.einsum("ci,ci->c", Wt, M) + np.log(pi + 1e-12)
    return dict(n, W2=Wt, b2=b)
for s in range(3):
    W = net(f"gelu_256_s{s}"); Nn = net(f"gelu_16_s{s}"); Hr = gl(Xr @ W["W1"].T + W["b1"]); HN = gl(Xr @ Nn["W1"].T + Nn["b1"])
    zs = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9); Cc = zs(Hr).T @ zs(HN) / len(Hr)
    print(f"seed {s}: trained 16: fine %.3f coarse %.3f KL %.3f" % score(Nn, W), "| trained reads + Bayes write: fine %.3f coarse %.3f KL %.3f" % score(bayes_write(W, Nn), W))
    parts = [np.argsort(-Cc[:, j])[:10] for j in range(16)]; print("   oracle top-10 groups + Bayes write: fine %.3f coarse %.3f KL %.3f" % score(bayes_write(W, build(W, parts)), W))
    for how in ("average", "ward", "random"):
        print(f"   {how:8s} co-firing groups + Bayes write: fine %.3f coarse %.3f KL %.3f" % score(bayes_write(W, build(W, groups_by(Hr, 16, how))), W))
