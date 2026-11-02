"""D2 look at the read scale: pre-activation std of trained narrow neurons vs their parts; built net vs a scale factor on
(u, b) after the rate-law threshold (description). 'parts' = rescale so the new vote std = parts' mean vote std."""
from d2_lib import *
W = net("gelu_256_s0"); Nn = net("gelu_16_s0"); Z = Xr @ W["W1"].T + W["b1"]; Hr = gl(Z); ZN = Xr @ Nn["W1"].T + Nn["b1"]
zs = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9); Cc = zs(Hr).T @ zs(gl(ZN)) / len(Hr); parts = [np.argsort(-Cc[:, j])[:10] for j in range(16)]
print("pre-activation std: trained narrow median %.2f; their parts median %.2f; wide median %.2f" % (np.median(ZN.std(0)), np.median([Z[:, p].std(0).mean() for p in parts]), np.median(Z.std(0))))
gs = groups_by(Hr, 16, "ward"); n0 = build_w(W, gs, [np.ones(len(g)) for g in gs]); S0 = Xr @ n0["W1"].T + n0["b1"]; print("built (unscaled) vote std median %.2f" % np.median(S0.std(0)))
tgt = np.array([Z[:, g].std(0).mean() for g in gs]) / S0.std(0)
for name, f in [("x0.25", .25), ("x0.5", .5), ("x1 (sum of z-votes)", 1.0), ("parts' std rule", tgt), ("x2", 2.0)]:
    f = np.broadcast_to(np.asarray(f, float), (16,)); n = dict(W1=n0["W1"] * f[:, None], b1=n0["b1"] * f)
    print(f"  scale {name:22s}: fine %.3f coarse %.3f KL %.3f" % score(bayes_write(W, n), W))
