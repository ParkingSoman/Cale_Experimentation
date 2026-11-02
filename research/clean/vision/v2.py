"""V2: cut L's MLP to 128 in every block (energy + LS refit, block by block) vs LN and S. Sealed accuracy."""
import numpy as np
from vlib import *

d = load_data(); ana, sealed = split_test(); cal = calib(d)
K = 128; F = f2c(d)
L = load_net("L", 128, 512, 25); ref = load_net("L", 128, 512, 25)
for b in range(6):
    U = stream_inputs(L, cal, b).reshape(-1, 128).astype(np.float64)   # inputs in the already-cut model
    W1, b1, W2, b2 = mlp_np(ref.blocks[b])
    Hh = gelu_np(U @ W1.T + b1); Y = Hh @ W2.T + b2                    # original MLP on these inputs
    keep = np.sort(np.argsort(-(Hh ** 2).mean(0) * (W2 ** 2).sum(0))[:K])
    W2n, b2n = ls_out(Hh[:, keep], Y)
    set_mlp(L.blocks[b], W1[keep], b1[keep], W2n, b2n)
    print("block", b, "rel err %.3f" % (np.linalg.norm(Hh[:, keep] @ W2n.T + b2n - Y) / np.linalg.norm(Y - Y.mean(0))), flush=True)
nets = {"L": ref, "L_cut": L, "LN": load_net("LN", 128, 128, 25), "S": load_net("S", 32, 128, 25), "L2": load_net("L2", 128, 512, 25)}
out, corr = {}, {}
for n, m in nets.items():
    lg = logits(m, d["xte"][sealed]); a = accuracy(lg, d["yte"][sealed], d["cte"][sealed], F)
    pred = lg.argmax(1); wrong = pred != d["yte"][sealed]
    within = float((F[pred[wrong]] == d["cte"][sealed][wrong]).mean())
    corr[n] = a; out[n] = {"fine": a["fine"], "super": a["super"], "own_error_within_super": within, "pred": pred}
yf = d["yte"][sealed]; yc = d["cte"][sealed]
newly = (out["L"]["pred"] == yf) & (out["L_cut"]["pred"] != yf)
out["L_cut"]["newly_wrong"] = int(newly.sum())
out["L_cut"]["newly_wrong_within_super"] = float((F[out["L_cut"]["pred"][newly]] == yc[newly]).mean())
out["Lcut_minus_LN"] = {k: boot_diff(corr["L_cut"][f"{k}_correct"], corr["LN"][f"{k}_correct"]) for k in ("fine", "super")}
for n in out:
    if isinstance(out[n], dict): out[n].pop("pred", None)
print(out); save("v2", out)
