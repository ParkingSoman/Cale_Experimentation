"""Statement 25.9 test: one natural-gradient (KFC-preconditioned) step from S with the gradient of the true second-order
objective from n = 2000 / 8000 / all train images, exact line search along each direction (curvature d^T F d from a
second pass over all train images). Developer-form formula (MIX-DERIVED, large sample); no knob."""
import numpy as np, torch, time
src = open("c17_fisher.py").read(); exec(src.split("G = torch.zeros(NRD, NRD")[0])
C = torch.load("cache/c17_fisher.pt"); Gf, Af = C["G"].to(DEV), C["A"].to(DEV)
Gi = torch.linalg.inv(Gf + 1e-6 * Gf.trace() / NRD * torch.eye(NRD, device=DEV)); Ai = torch.linalg.inv(Af + 1e-6 * Af.trace() / 962 * torch.eye(962, device=DEV))
Pinv = lambda R: (Gi @ (R * free) @ Ai) * free
vS = Vs["S"] * mask; allimg = np.random.default_rng(21).permutation(D.tr); N = len(allimg); marks = {2000: None, 8000: None}
g = torch.zeros(NRD, 962, device=DEV); L0 = 0.0
for i in range(0, N, 10):
    Gm, _, _ = gammas(allimg[i:i + 10]); sc = (Gm * vS).sum((2, 3)); g += torch.einsum("bc,bcjk->jk", sc, Gm); L0 += float((sc ** 2).sum()) / 2
    if i + 10 in marks: marks[i + 10] = (g / (i + 10)).clone()
    if i % 5000 == 0: print(f"  pass 1: {i} images ({time.time() - t0:.0f}s)", flush=True)
marks[N] = g / N; L0 /= N
dirs = {n: -Pinv(gn * free) for n, gn in marks.items()}
curv = {n: 0.0 for n in dirs}
for i in range(0, N, 10):
    Gm, _, _ = gammas(allimg[i:i + 10])
    for n, d in dirs.items(): curv[n] += float(((Gm * d).sum((2, 3)) ** 2).sum())
    if i % 5000 == 0: print(f"  pass 2: {i} images ({time.time() - t0:.0f}s)", flush=True)
def build(V):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m); off = np.cumsum([0] + sizes)
    with torch.no_grad():
        for cc in fm.cols: cc.zero_()
        for j in range(4):
            o = 0 if j < 3 else 481; Bj = -V[off[j]:off[j + 1], (o + restt.to(DEV))]
            fm.dW[j].copy_(Bj[:, :470] / fm.cscale[j]); fm.cbp[j].copy_(Bj[:, 470] / 0.1)
    return fm.to(DEV)
g_all = marks[N] * free; res = {}; refs = REFS(); g5 = np.load("cache/c5_gapb.npz"); refs["T"] = g5["T"]
print(f"population objective at S: {L0:.5f}; Newton decrement under KFC: {0.5 * float((g_all * -dirs[N]).sum()):.5f}")
for n, d in dirs.items():
    a = -float((g_all * d).sum()) / (curv[n] / N); pred = 0.5 * a * -float((g_all * d).sum())
    cos = float((dirs[n] * dirs[N]).sum() / dirs[n].norm() / dirs[N].norm())
    fm = build(vS + a * d); torch.save(fm.state_dict(), f"cache/c21_step_n{n}.pt"); r = per_image(fm); res[str(n)] = r
    print(f"n = {n:5d}: cos with population direction {cos:.3f}, step {a:.3f}, predicted decrease {pred:.5f}; dev-5000 KL {r.mean():.4f} (se {r.std() / np.sqrt(len(r)):.4f})", flush=True)
    paired(f"step_n{n}", r, {"kbar(S)": refs["kbar"], "T": refs["T"]})
np.savez("cache/c21_newton.npz", **res); print(f"done ({time.time() - t0:.0f}s)")
