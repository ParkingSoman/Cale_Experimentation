"""F4 (microscope): preconditioned CG on the true second-order objective 1/2 v^T F_n v over the free reader coordinates,
start = S (factorised build), F_n from n stored per-image (g, y); preconditioner plain KFC; noise-line stopping rule."""
import numpy as np, torch, time, sys
src = open("c17_fisher.py").read(); exec(src.split("G = torch.zeros(NRD, NRD")[0])
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000; imgs = TR[:n]
gs, ys = [], []
for i in range(0, n, 10):
    b = imgs[i:i + 10]; gs.append(reader_grads(b).reshape(len(b), 10, NRD, 196).half()); ys.append(ybatch(b).half())
gs, ys = torch.cat(gs), torch.cat(ys); print(f"stored g {tuple(gs.shape)}, y {tuple(ys.shape)} ({time.time() - t0:.0f}s)", flush=True)
def per_image_scores(V, bs=50):                                    # s[x, c] = <Gamma_c(x), V>
    out = []
    for i in range(0, n, bs):
        VY = torch.einsum("jk,bkp->bjp", V, ys[i:i + bs].float()); out.append(torch.einsum("bcjp,bjp->bc", gs[i:i + bs].float(), VY))
    return torch.cat(out)
def back(Sc, bs=50):                                               # sum_x sum_c Sc[x,c] Gamma_c(x) / n
    R = torch.zeros(NRD, 962, device=DEV)
    for i in range(0, n, bs):
        W = torch.einsum("bc,bcjp->bjp", Sc[i:i + bs], gs[i:i + bs].float()); R += torch.einsum("bjp,bkp->jk", W, ys[i:i + bs].float())
    return R * mask / n
Fv = lambda V: back(per_image_scores(V))
C = torch.load("cache/c17_fisher.pt"); Gf, Af = C["G"].to(DEV), C["A"].to(DEV)
Gi = torch.linalg.inv(Gf + 1e-6 * Gf.trace() / NRD * torch.eye(NRD, device=DEV)); Ai = torch.linalg.inv(Af + 1e-6 * Af.trace() / 962 * torch.eye(962, device=DEV))
Pinv = lambda R: (Gi @ (R * free) @ Ai) * free
v = Vs["S"] * mask
# per-image gradient noise at S for the stopping rule: g_x = sum_c s_xc Gamma_c(x)
def grad_and_noise(V, bs=25):
    Sc = per_image_scores(V); gsum = torch.zeros(NRD, 962, device=DEV); tr = 0.0
    for i in range(0, n, bs):
        for x in range(i, min(i + bs, n)):
            gx = torch.einsum("c,cjp,kp->jk", Sc[x], gs[x].float(), ys[x].float()) * free
            gsum += gx; tr += float((gx * Pinv(gx)).sum())
    g = gsum / n; noise = (tr / n - float((g * Pinv(g)).sum())) / n
    return g, noise
def build(V):
    fm = b3_train.FamilyRA(m, fam, 1); fm.add_rb(m); fm.add_ra(m); off = np.cumsum([0] + sizes)
    with torch.no_grad():
        for cc in fm.cols: cc.zero_()
        for j in range(4):
            o = 0 if j < 3 else 481; Bj = -V[off[j]:off[j + 1], (o + restt.to(DEV))]
            fm.dW[j].copy_(Bj[:, :470] / fm.cscale[j]); fm.cbp[j].copy_(Bj[:, 470] / 0.1)
    return fm.to(DEV)
g, noise = grad_and_noise(v); sig = float((g * Pinv(g)).sum())
print(f"start: |g|_P^2 {sig:.3e}, noise line {noise:.3e}, ratio {sig / noise:.1f}; L {0.5 * float((v * Fv(v)).sum()):.5f} ({time.time() - t0:.0f}s)", flush=True)
r = -g; z = Pinv(r); p = z.clone(); rz = float((r * z).sum()); stop = None; kls = {}
for k in range(1, 31):
    Fp = Fv(p); al = rz / float((p * Fp).sum()); v = v + al * p; r = r - al * Fp
    L = 0.5 * float((v * Fv(v)).sum())
    if k in (1, 2, 3, 5, 8, 12, 20, 30):
        res = per_image(build(v)); kls[k] = res; print(f"  it {k}: L {L:.5f}  dev-5000 KL {res.mean():.4f} (se {res.std() / np.sqrt(len(res)):.4f})", flush=True)
        torch.save(v.cpu(), f"cache/c19_v_it{k}.pt")
    if stop is None and k % 2 == 0:
        gk, nk = grad_and_noise(v); sk = float((gk * Pinv(gk)).sum())
        print(f"  it {k}: |g|_P^2 {sk:.3e} vs noise {nk:.3e}", flush=True)
        if sk < nk: stop = k; print(f"  STOP RULE fires at it {k}", flush=True)
    z = Pinv(r); rz2 = float((r * z).sum()); p = z + (rz2 / rz) * p; rz = rz2
    if stop is not None and k >= max(stop, 12): break
np.savez("cache/c19_pcg.npz", **{f"it{k}": v_ for k, v_ in kls.items()}); print(f"stop {stop}; done ({time.time() - t0:.0f}s)")
