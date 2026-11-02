from n9_kfac import kfac_stats
from n7_lib import *
f = load("16_2886"); th = dict(f.parent); C = len(th["u"]); kv0 = f.trainkl(th); s = kfac_stats(f, th)
A, bv = gn_stats(f, th, f.gix, S=2)
gw_full = bv[C + 1:]; print("grad w match:", cos(gw_full, s["gw"]), "grad u match:", cos(bv[:C], s["gu"]))
Aw_full = A[C + 1:, C + 1:]; Aw_k = s["G"] * s["hh2_mean"]
print("trace Aw full %.3e  kfac %.3e ; ratio %.2f" % (np.trace(Aw_full), np.trace(Aw_k), np.trace(Aw_full) / np.trace(Aw_k)))
Au_full = A[:C, :C]; Au_k = s["Aa"] * s["ee2"]; print("trace Au full %.3e kfac %.3e ratio %.2f" % (np.trace(Au_full), np.trace(Au_k), np.trace(Au_full) / np.trace(Au_k)))
for lm in (1e-1, 1, 10, 100, 1e3):
    dw = -np.linalg.solve(Aw_k + lm * np.diag(np.diag(Aw_k)), s["gw"]); dwf = -np.linalg.solve(Aw_full + lm * np.diag(np.diag(Aw_full)), gw_full)
    for al in (.25, 1):
        print(lm, al, "w-only kfac %.4f full %.4f (base %.4f)  |dw| %.3f vs %.3f cos %.2f" % (f.trainkl({**th, "w": th["w"] + al * dw}) * 1e4, f.trainkl({**th, "w": th["w"] + al * dwf}) * 1e4, kv0 * 1e4, np.linalg.norm(al * dw), np.linalg.norm(al * dwf), cos(dw, dwf)))
