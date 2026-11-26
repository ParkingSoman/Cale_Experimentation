"""Review checks for N2/N3 (2026-11-13).
1. Where is the removal loss? Split each recipe's per-photo KL into photos where a removed child fires strongly
   (peak at or above that child's 500th largest test peak) and the rest. Also: zero fill against mean fill.
2. Controls for the coarse child (A3all + bias): each child keeps the same NUMBER of inputs as in A3all, chosen
   (a) at random, (b) as the largest by mean |part| at the child's peak over all fit photos (no clustering).
   Bootstrap each against drop + bias.
"""
import json
import os

import numpy as np
import torch

from absorb import net, DEV, BATCH, OUT

RECIPES = {245: [243, 73, 377], 418: [317, 408]}
KIDS = [243, 73, 377, 317, 408]


def main():
    torch.set_default_dtype(torch.float32)
    mix = json.load(open(os.path.join(OUT, "mix.json")))
    cache = np.load(os.path.join(OUT, "absorb_cache.npz"))
    m = net.load().to(DEV)
    ph = net.Photos("read")
    lab, fit = cache["lab"], cache["fit"]
    means = torch.tensor(cache["means"]).to(DEV)
    h_mean = net.block_means(os.path.join(net.STORE, "read"))[4][0].to(DEV)
    b0, b1 = m.layer4[0], m.layer4[1]
    sc = b1.bn1.weight / torch.sqrt(b1.bn1.running_var + b1.bn1.eps)
    W1 = b1.conv1.weight[KIDS]
    mean_map = h_mean.view(1, -1, 1, 1).expand(1, 512, 7, 7).contiguous()
    mean_pad = torch.nn.functional.pad(mean_map, (1, 1, 1, 1))
    N = len(lab)
    peak = np.zeros((N, len(KIDS)), np.float32)
    absum = torch.zeros(len(KIDS), 512, device=DEV)
    nfit = 0
    g_zero = {p: np.zeros((N, 512), np.float32) for p in RECIPES}

    def run_inner(h40, keep=None, fill=None, kids=None):
        u = b1.bn1(b1.conv1(h40))
        if keep:
            for j, chans in keep.items():
                mask = torch.zeros(512, device=DEV)
                mask[chans] = 1
                u[:, j] = b1.bn1(b1.conv1(mean_map + (h40 - mean_map) * mask.view(1, -1, 1, 1)))[:, j]
        inner = torch.relu(u)
        if fill is not None:
            inner = inner.clone()
            inner[:, kids] = fill
        return u, torch.relu(b1.bn2(b1.conv2(inner)) + h40).mean((2, 3))

    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl = slice(i, min(i + BATCH, N))
            idx = np.arange(sl.start, sl.stop)
            x = ph.tensor(idx).float().to(DEV)
            h3 = m.layer3(m.layer2(m.layer1(net.stem(m, x))))
            h40 = net.block_parts(b0, h3)["h"]
            u, _ = run_inner(h40)
            uk = u[:, KIDS].flatten(2)
            mx, am = uk.max(2)
            peak[sl] = mx.cpu().numpy()
            f = torch.as_tensor(fit[idx], device=DEV)
            if f.any():
                dh = torch.nn.functional.pad(h40, (1, 1, 1, 1)) - mean_pad
                for k, j in enumerate(KIDS):
                    r, q = am[:, k] // 7, am[:, k] % 7
                    win = torch.stack([dh[b, :, r[b]:r[b] + 3, q[b]:q[b] + 3] for b in range(len(idx))])
                    part = (win * W1[k].unsqueeze(0)).sum((2, 3)) * sc[j]
                    absum[k] += part[f].abs().sum(0)
                nfit += int(f.sum())
            for p, kids in RECIPES.items():
                _, g = run_inner(h40, fill=torch.zeros(1, len(kids), 1, 1, device=DEV), kids=kids)
                g_zero[p][sl] = g.cpu().numpy()
    print("pass 1 done", flush=True)
    rng = np.random.default_rng(3)
    keep_all = {j: sorted(set(t for s in mix[str(j)]["subcircuits"] for t in s["top10"])) for j in KIDS}
    top_abs = {j: np.argsort(-(absum[k] / nfit).cpu().numpy())[:len(keep_all[j])].tolist() for k, j in enumerate(KIDS)}
    rnd = {j: sorted(rng.choice(512, len(keep_all[j]), replace=False).tolist()) for j in KIDS}
    g_ctl = {f"{p}_{v}": np.zeros((N, 512), np.float32) for p in RECIPES for v in ("top_abs", "random")}
    with torch.no_grad():
        for i in range(0, N, BATCH):
            sl = slice(i, min(i + BATCH, N))
            x = ph.tensor(np.arange(sl.start, sl.stop)).float().to(DEV)
            h3 = m.layer3(m.layer2(m.layer1(net.stem(m, x))))
            h40 = net.block_parts(b0, h3)["h"]
            for p, kids in RECIPES.items():
                for v, ks in (("top_abs", top_abs), ("random", rnd)):
                    _, g = run_inner(h40, keep={j: ks[j] for j in kids})
                    g_ctl[f"{p}_{v}"][sl] = g.cpu().numpy()
    print("pass 2 done", flush=True)

    torch.set_default_dtype(torch.float64)
    from controls import solve, boot
    from fit import kl_rows
    W, bias = torch.tensor(cache["fcW"]).double(), torch.tensor(cache["fcb"]).double()
    yN = torch.tensor(cache["gN"]).double() @ W.T + bias
    a2 = np.load(os.path.join(OUT, "absorb2_cache.npz"))
    F, T = np.where(fit)[0], np.where(~fit)[0]
    res = dict(overlap_top_abs_vs_all={j: len(set(top_abs[j]) & set(keep_all[j])) / len(keep_all[j]) for j in KIDS}, rows=[])
    for p, kids in RECIPES.items():
        ks = [KIDS.index(j) for j in kids]
        thr = [np.sort(peak[T, k])[-500] for k in ks]
        strong = np.zeros(len(T), bool)
        for k, t in zip(ks, thr):
            strong |= peak[T, k] >= t
        y = lambda g: torch.tensor(g).double() @ W.T + bias
        yD = y(a2[f"{p}_drop"])
        kd = kl_rows(yN[T], yD[T]).numpy()
        yZ = y(g_zero[p])
        kz = kl_rows(yN[T], yZ[T]).numpy()
        row = dict(parent=p, n_strong=int(strong.sum()), n_test=len(T),
                   drop_mean_fill=float(kd.mean()), share_on_strong=float(kd[strong].sum() / kd.sum()),
                   drop_zero_fill=float(kz.mean()), zero_share_on_strong=float(kz[strong].sum() / kz.sum()))
        a, _ = solve(yN[F], yD[F], None)
        base = kl_rows(yN[T], yD[T] + a)
        row["drop_bias"] = float(base.mean())
        for v, g in (("A3all", a2[f"{p}_A3all"]), ("top_abs", g_ctl[f"{p}_top_abs"]), ("random", g_ctl[f"{p}_random"])):
            yv = y(g)
            a, _ = solve(yN[F], yv[F], None)
            t = kl_rows(yN[T], yv[T] + a)
            row[v + "_bias"] = float(t.mean())
            row[v + "_minus_dropbias"] = boot(t - base)
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
    print("overlap top_abs vs A3all inputs:", res["overlap_top_abs_vs_all"], flush=True)
    json.dump(res, open(os.path.join(OUT, "link.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
