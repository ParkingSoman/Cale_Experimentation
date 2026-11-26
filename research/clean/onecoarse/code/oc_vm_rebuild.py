# scale: neurons=s3:69, s3:all, A:children, A:block0-all, A:block1-weights, A:head-weights; net=A; photos=held, test; where=cluster
# rung: Q=vector-math track step 3: rebuild the 11 deleted children from ALL kept firings, beta by stream size (known method) vs by KL (square rule); N=1 deleted set of 11, 758 carriers; EST=30 min CPU; PREV=results/oc_dial.npz, results/oc_when.npz; PERMIT=Shonu approved plan "New track: deletion as vector math" 2026-11-24 incl. its SCALE-UP REQUEST
"""VECMATH step 3 (2026-11-24). Calculated builds only, nothing trained.

Network A, stage-4 block 0. D = the 11 R1 children of s3:69. Delete D: their firings a_d are set to 0.
Carriers (every kept unit, nothing picked by hand), 758 firing signals F_j per cell of the 7 x 7 map:
  j < 501      kept block-0 neuron firings (inner = ReLU(bn1(conv1 x)))
  501..756     stage-3 channels at the 1x1 skip's positions (x[:, :, ::2, ::2])
  757          the constant 1
Rebuild (one operation): a_hat_d = sum_j beta_dj F_j. Kept weight slots that carry it:
  block-0 carrier c: its conv2 write column += sum_d beta_dc V_d (3x3, bn2 scale folded)
  skip carrier k:    its 1x1 skip column     += sum_d beta_dk u_d, u_d = V_d summed over the 3x3 offsets
  constant:          the bn2 bias            += sum_d beta_d,const u_d
So the stream change is conv(a_hat_blk, V_D) + 1x1(a_hat_skip, U_D). No weight of D survives.
beta:
  V-known  plain least squares of a_d on F over DEV cells (the interpolative-decomposition / ZipLM-style fold)
  V-KL     the lowest second-order KL: KL(A || build) ~ 1/2 sum_k p_k (dz_k - p.dz)^2, dz = J e, e = stream error.
           Rows G_k = sqrt(p_k) (J_k - sum_j p_j J_j) for the top M classes of each photo (mass reported).
           Normal equations N beta = h are formed exactly over DEV and solved directly (damping picked on a DEV split).
Outputs: KL per photo of V0 / V-known / V-KL on test sets, predicted KL (square rule), carrier shares, a stop curve
over the top-k carriers, a curve over the number of DEV photos.
Usage: python -u oc_vm_rebuild.py [smoke]
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import r18
from capture18 import net_for

P = 69
KIDS = [305, 97, 23, 163, 385, 297, 464, 237, 108, 510, 275]
SMOKE = "smoke" in sys.argv
DEV = r18.DEV
M_TOP = 16
BS = 125
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))


def bn_parts(bn):
    sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return sc, bn.bias - bn.running_mean * sc


def main():
    t0 = time.time()
    m = net_for("A").to(DEV).eval()
    for q in m.parameters():
        q.requires_grad_(False)
    b0, b1 = m.layer4[0], m.layer4[1]
    s1, _ = bn_parts(b0.bn1)
    s2, _ = bn_parts(b0.bn2)
    R = b0.conv1.weight * s1.view(-1, 1, 1, 1)
    V = b0.conv2.weight * s2.view(-1, 1, 1, 1)                                   # 512 out x 512 in x 3 x 3
    wnorm = torch.sqrt((V ** 2).sum((0, 2, 3)))
    order = np.argsort(-(R[:, P].sum((1, 2)) * wnorm).cpu().numpy())
    assert order[:11].tolist() == KIDS, order[:11]
    kt = torch.as_tensor(KIDS, device=DEV)
    kept = torch.as_tensor(np.setdiff1d(np.arange(512), KIDS), device=DEV)      # 501
    VD = V[:, kt].contiguous()                                                   # 512 x 11 x 3 x 3
    UD = VD.sum((2, 3))                                                          # 512 x 11 (1x1 collapse)
    NB, NS = 501, 256
    NJ = NB + NS + 1
    vnorm2 = (VD ** 2).sum((0, 2, 3)).double().cpu().numpy()                     # |V_d|^2

    held, test = r18.Cache("held"), r18.Cache("test")
    dev_rows = np.arange(len(held))
    rng = np.random.default_rng(7)
    perm = rng.permutation(dev_rows)                                              # chunks for the size curve
    te_rand = np.sort(np.random.default_rng(2).choice(len(test), 2000, replace=False))

    def parts(x):
        pre = b0.bn1(b0.conv1(x))
        inner = torch.relu(pre)
        Fb = inner[:, kept]
        Fs = x[:, :, ::2, ::2]
        one = torch.ones_like(Fs[:, :1])
        return inner, torch.cat([Fb, Fs, one], 1)                                # B x 758 x 7 x 7

    def tail(x, r):
        return r18.net.head(m, r18.net.block_parts(b1, torch.relu(r + b0.downsample(x)))["h"])

    def stream(x, inner, beta=None, Fall=None):
        """block-0 output r (before the skip): kept neurons + bn2 bias + rebuild."""
        inner = inner.clone()
        inner[:, kt] = 0
        r = b0.bn2(b0.conv2(inner))
        if beta is not None:
            bt = torch.as_tensor(beta, dtype=torch.float32, device=DEV)           # 11 x 758
            ah_b = torch.einsum("dj,bjhw->bdhw", bt[:, :NB], Fall[:, :NB])
            ah_s = torch.einsum("dj,bjhw->bdhw", bt[:, NB:], Fall[:, NB:])
            r = r + F.conv2d(ah_b, VD, padding=1) + torch.einsum("od,bdhw->bohw", UD, ah_s)
        return r

    # D's summed firing per photo (for the photo groups), on DEV and all of test
    @torch.no_grad()
    def dsum(cache, rows):
        out = []
        for s in range(0, len(rows), 500):
            x, _ = cache.get(rows[s:s + 500])
            out.append(torch.relu(b0.bn1(b0.conv1(x)))[:, kt].sum((1, 2, 3)).cpu().numpy())
        return np.concatenate(out)

    te_all = np.arange(len(test))
    dev_d = dsum(held, dev_rows)
    te_d = dsum(test, te_all)
    hi, lo = np.quantile(dev_d, .9), np.quantile(dev_d, .5)
    pk = []
    for s in range(0, len(test), 1000):
        xs = test.x[s:s + 1000, P]
        xs = xs if torch.is_tensor(xs) else torch.as_tensor(np.ascontiguousarray(xs))
        pk.append(xs.float().amax((1, 2)).cpu().numpy())
    te_strong = np.nonzero(np.concatenate(pk) >= 2.92)[0]
    te_fired = np.nonzero(te_d >= hi)[0]
    te_quiet = np.intersect1d(te_rand, np.nonzero(te_d <= lo)[0])
    if SMOKE:
        dev_rows, perm = dev_rows[:64], perm[:64]
        te_rand, te_strong, te_fired, te_quiet = te_rand[:48], te_strong[:48], te_fired[:48], te_quiet[:48]
    print(f"test {len(test)}: strong {len(te_strong)}, D fired most {len(te_fired)} (>= {hi:.1f}), "
          f"D barely fired {len(te_quiet)} (<= {lo:.1f}, inside te_rand); {time.time()-t0:.0f} s", flush=True)

    # ---- DEV pass: plain moments and the KL normal equations, per chunk of the permuted DEV order
    @torch.enable_grad()
    def g_rows(x, r, z):
        """Rows of the square rule at A: B x M x 512 x 7 x 7, and the class mass covered."""
        p = F.softmax(z.float(), 1)
        top = p.topk(M_TOP, 1)
        rr = r.detach().requires_grad_()
        y = tail(x, rr)
        gm = torch.autograd.grad((y * p).sum(), rr, retain_graph=True)[0]          # J^T p, per photo
        rows = []
        for k in range(M_TOP):
            sel = y.gather(1, top.indices[:, k:k + 1]).sum()
            gk = torch.autograd.grad(sel, rr, retain_graph=k < M_TOP - 1)[0]
            rows.append(top.values[:, k].sqrt().view(-1, 1, 1, 1) * (gk - gm))
        return torch.stack(rows, 1), top.values.sum(1)

    def gphi(G, Fall):
        """G Phi: B x M x 11 x 758, and the adjoint maps."""
        B = G.shape[0]
        Gf = G.reshape(B * M_TOP, 512, 7, 7)
        Wb = F.conv_transpose2d(Gf, VD, padding=1).reshape(B, M_TOP, 11, 7, 7)   # d/d(a_hat_blk)
        Ws = torch.einsum("bmohw,od->bmdhw", G, UD)                               # d/d(a_hat_skip)
        out = torch.cat([torch.einsum("bmdhw,bjhw->bmdj", Wb, Fall[:, :NB]),
                         torch.einsum("bmdhw,bjhw->bmdj", Ws, Fall[:, NB:])], 3)
        return out

    sizes = [250, 500, 1000, 2000] if not SMOKE else [16, 32, 64]
    bounds = [0] + sizes
    acc = {s: dict(FF=np.zeros((NJ, NJ)), FA=np.zeros((NJ, 11)), AA=np.zeros(11), N=np.zeros((11 * NJ, 11 * NJ)),
                   h=np.zeros(11 * NJ), c=0.0, cells=0, mass=0.0, n=0) for s in sizes}
    for ci in range(len(sizes)):
        rows_c = perm[bounds[ci]:bounds[ci + 1]]
        part = dict(FF=np.zeros((NJ, NJ)), FA=np.zeros((NJ, 11)), AA=np.zeros(11), N=np.zeros((11 * NJ, 11 * NJ)),
                    h=np.zeros(11 * NJ), c=0.0, cells=0, mass=0.0, n=0)
        for s in range(0, len(rows_c), BS):
            rows = np.sort(rows_c[s:s + BS])
            x, z = held.get(rows)
            with torch.no_grad():
                inner, Fall = parts(x)
                aD = inner[:, kt]
                L = F.conv2d(aD, VD, padding=1)                                   # what D wrote (to be rebuilt)
                r0 = stream(x, inner)                                             # V0 stream
                Ff = Fall.permute(0, 2, 3, 1).reshape(-1, NJ).double()
                Af = aD.permute(0, 2, 3, 1).reshape(-1, 11).double()
                part["FF"] += (Ff.T @ Ff).cpu().numpy(); part["FA"] += (Ff.T @ Af).cpu().numpy()
                part["AA"] += (Af ** 2).sum(0).cpu().numpy(); part["cells"] += Ff.shape[0]
            G, mass = g_rows(x, r0 + L, z)                                        # rows at A's own stream
            with torch.no_grad():
                GP = gphi(G, Fall).reshape(len(rows) * M_TOP, -1)                 # (B*M) x (11*758), float32
                GL = (G * L.unsqueeze(1)).sum((2, 3, 4)).reshape(-1)              # B*M
                part["N"] += (GP.T @ GP).double().cpu().numpy()
                part["h"] += (GP.T @ GL).double().cpu().numpy()
                GL = GL.double()
                part["c"] += float((GL ** 2).sum()); part["mass"] += float(mass.sum()); part["n"] += len(rows)
        for s_ in sizes[ci:]:
            for k_ in part:
                acc[s_][k_] = acc[s_][k_] + part[k_]
        print(f"DEV chunk {ci} ({len(rows_c)} photos) done; {time.time()-t0:.0f} s", flush=True)
    full = acc[sizes[-1]]
    print(f"top-{M_TOP} class mass on DEV: {full['mass'] / full['n']:.4f}", flush=True)

    # ---- solves
    def solve_known(a, lam_rel, S=None):
        S = np.arange(NJ) if S is None else S
        A_ = a["FF"][np.ix_(S, S)]
        lam = lam_rel * np.trace(A_) / len(S)
        b = np.linalg.solve(A_ + lam * np.eye(len(S)), a["FA"][S])                # |S| x 11
        beta = np.zeros((11, NJ)); beta[:, S] = b.T
        return beta

    def solve_kl(a, lam_rel, S=None):
        S = np.arange(NJ) if S is None else S
        idx = (np.arange(11)[:, None] * NJ + S[None, :]).ravel()
        N_ = a["N"][np.ix_(idx, idx)]
        lam = lam_rel * np.trace(N_) / len(idx)
        b = np.linalg.solve(N_ + lam * np.eye(len(idx)), a["h"][idx])
        beta = np.zeros((11, NJ)); beta[:, S] = b.reshape(11, len(S))
        return beta

    def quad(a, beta):                                                            # 1/2 |G Phi beta - G L|^2, summed
        v = beta.ravel()
        return .5 * (v @ a["N"] @ v - 2 * v @ a["h"] + a["c"])

    def known_resid(a, beta):
        Bt = beta.T
        return float(np.trace(Bt.T @ a["FF"] @ Bt) - 2 * np.trace(Bt.T @ a["FA"]) + a["AA"].sum())

    # damping on a DEV split: fit on the first 3 chunks (1,000 photos), check on the last chunk (1,000 photos)
    fit_a = acc[sizes[-2]]
    val = {k_: full[k_] - fit_a[k_] for k_ in ("FF", "FA", "AA", "N", "h", "c")}
    lams = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, .3, 1., 3., 10., 30., 100.]   # widened 2026-11-24: KL pick sat at the old edge (.1)
    vk = [known_resid(val, solve_known(fit_a, l)) for l in lams]
    vq = [quad(val, solve_kl(fit_a, l)) for l in lams]
    lam_k, lam_q = lams[int(np.argmin(vk))], lams[int(np.argmin(vq))]
    print("damping check (held-out DEV half): known", np.round(vk, 2).tolist(), "-> ", lam_k,
          "| KL quad", np.round(vq, 5).tolist(), "->", lam_q, flush=True)
    beta_k = solve_known(full, lam_k)
    beta_q = solve_kl(full, lam_q)

    # carrier shares: share_j = sum_d |V_d|^2 beta_dj E[F_j a_hat_d] / sum_d |V_d|^2 E[a_hat_d^2]
    def shares(beta, a):
        FF = a["FF"]
        num = (vnorm2[:, None] * beta * (beta @ FF)).sum(0)                       # per carrier
        den = (vnorm2 * np.einsum("dj,jk,dk->d", beta, FF, beta)).sum()
        per_d = beta * (beta @ FF) / np.einsum("dj,jk,dk->d", beta, FF, beta)[:, None]
        return num / den, per_d
    sh_k, pd_k = shares(beta_k, full)
    sh_q, pd_q = shares(beta_q, full)
    names = [f"b0:n{int(c)}" for c in kept.cpu().numpy()] + [f"s3:{k}" for k in range(NS)] + ["const"]
    for nm, sh in (("V-known", sh_k), ("V-KL", sh_q)):
        top = np.argsort(-sh)[:12]
        print(f"{nm} top carriers:", " ".join(f"{names[j]} {sh[j]:.3f}" for j in top), flush=True)

    # stop curve over the top-k carriers (order from V-KL's shares) and the size curve
    ordk = np.argsort(-sh_q)
    ks = [1, 3, 10, 30, 100, 300, NJ] if not SMOKE else [1, 10, NJ]
    builds = {"V-known": beta_k, "V-KL": beta_q}
    for k in ks[:-1]:
        S = np.sort(ordk[:k])
        builds[f"known_k{k}"] = solve_known(full, lam_k, S)
        builds[f"kl_k{k}"] = solve_kl(full, lam_q, S)
    for s_ in sizes[:-1]:
        builds[f"known_n{s_}"] = solve_known(acc[s_], lam_k)
        builds[f"kl_n{s_}"] = solve_kl(acc[s_], lam_q)

    # ---- evaluation on test sets: real KL, and the square rule's prediction for the main builds
    sets = {"te_rand": te_rand, "te_strong": te_strong, "te_fired": te_fired, "te_quiet": te_quiet}
    evl = np.unique(np.concatenate(list(sets.values())))
    res = dict(kids=np.array(KIDS), names=np.array(names), beta_known=beta_k, beta_kl=beta_q, share_known=sh_k,
               share_kl=sh_q, perd_known=pd_k, perd_kl=pd_q, lam_known=lam_k, lam_kl=lam_q, val_known=np.array(vk),
               val_kl=np.array(vq), ks=np.array(ks), sizes=np.array(sizes), dev_d=dev_d, te_d=te_d, hi=hi, lo=lo,
               dev_mass=full["mass"] / full["n"], evl=evl, **{f"set_{k}": v for k, v in sets.items()})
    kl = {b: [] for b in ["base", "V0"] + list(builds)}
    pred = {b: [] for b in ("V0", "V-known", "V-KL")}
    for s in range(0, len(evl), BS):
        rows = evl[s:s + BS]
        x, z = test.get(rows)
        with torch.no_grad():
            inner, Fall = parts(x)
            L = F.conv2d(inner[:, kt], VD, padding=1)
            r0 = stream(x, inner)
            kl["base"].append(r18.kl_rows(z, tail(x, r0 + L)).cpu().numpy())
            kl["V0"].append(r18.kl_rows(z, tail(x, r0)).cpu().numpy())
            for b, beta in builds.items():
                kl[b].append(r18.kl_rows(z, tail(x, stream(x, inner, beta, Fall))).cpu().numpy())
        G, _ = g_rows(x, r0 + L, z)
        with torch.no_grad():
            GP = gphi(G, Fall).reshape(len(rows), M_TOP, -1).double()
            GL = (G * L.unsqueeze(1)).sum((2, 3, 4)).double()
            for b, beta in (("V0", np.zeros((11, NJ))), ("V-known", beta_k), ("V-KL", beta_q)):
                v = torch.as_tensor(beta.ravel(), device=GP.device)
                pred[b].append((.5 * ((GP @ v) - GL) ** 2).sum(1).cpu().numpy())
    kl = {b: np.concatenate(v) for b, v in kl.items()}
    pred = {b: np.concatenate(v) for b, v in pred.items()}
    assert kl["base"].max() < 1e-4, f"base KL {kl['base'].max():.2e}"
    pos = {r_: i for i, r_ in enumerate(evl)}
    for b, v in kl.items():
        res[f"kl_{b}"] = v
    for b, v in pred.items():
        res[f"pred_{b}"] = v
    print(f"base KL max {kl['base'].max():.1e}", flush=True)
    for sn, rows in sets.items():
        ix = np.array([pos[r_] for r_ in rows])
        d0 = kl["V0"][ix].mean()
        line = [f"[{sn}] n {len(rows)}, V0 {d0:.5f}"]
        for b in ("V-known", "V-KL"):
            rec = 1 - kl[b][ix].mean() / d0
            prec = 1 - pred[b][ix].mean() / pred["V0"][ix].mean()
            line.append(f"{b} rec {rec:.3f} (square-rule pred {prec:.3f})")
        print("; ".join(line), flush=True)
    for sn in ("te_fired", "te_quiet", "te_strong", "te_rand"):
        ix = np.array([pos[r_] for r_ in sets[sn]])
        d0 = kl["V0"][ix].mean()
        print(f"  {sn} curve k:", " ".join(f"{k}:{1 - kl[f'known_k{k}'][ix].mean()/d0:.2f}/{1 - kl[f'kl_k{k}'][ix].mean()/d0:.2f}"
                                           for k in ks[:-1]), "| n:",
              " ".join(f"{s_}:{1 - kl[f'known_n{s_}'][ix].mean()/d0:.2f}/{1 - kl[f'kl_n{s_}'][ix].mean()/d0:.2f}"
                       for s_ in sizes[:-1]), flush=True)

    out = os.path.join(r18.OUT, "oc_vm_rebuild" + ("_smoke" if SMOKE else "") + ("_wide" if "wide" in sys.argv else "") + ".npz")
    np.savez(out, **res)
    print("saved", out, f"{time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
