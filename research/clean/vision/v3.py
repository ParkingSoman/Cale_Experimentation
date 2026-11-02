"""V3: true (label) families vs cut vs random vs Ward merges, one block of L cut to 128. Sealed accuracy."""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import kmeans2
from vlib import *

d = load_data(); ana, sealed = split_test(); cal = calib(d); F = f2c(d); K = 128
yf_a, yc_a = d["yte"][ana], d["cte"][ana]
res = {}
for b in (5, 3):
    m = load_net("L", 128, 512, 25)
    A = neuron_max(m, d["xte"][ana], b); top = np.argsort(-A, 0)[:100].T
    fam = np.array([np.bincount(yc_a[t], minlength=20).argmax() for t in top])
    hist = np.stack([np.bincount(yf_a[t], minlength=100) / 100 for t in top])
    n_s = np.bincount(fam, minlength=20)
    k_s = np.floor(K * n_s / 512).astype(int); k_s[n_s > 0] = np.maximum(k_s[n_s > 0], 1)
    while k_s.sum() < K: k_s[np.argmax(n_s / np.maximum(k_s, 1) * (k_s < n_s))] += 1
    while k_s.sum() > K: k_s[np.argmax(np.where(k_s > 1, k_s, 0))] -= 1
    g = np.zeros(512, int); nxt = 0
    for s in range(20):
        idx = np.where(fam == s)[0]
        if len(idx) == 0: continue
        if k_s[s] >= len(idx): lab = np.arange(len(idx))
        else:
            _, lab = kmeans2(hist[idx], k_s[s], seed=0, minit="++")
            lab = np.unique(lab, return_inverse=True)[1]
        g[idx] = lab + nxt; nxt += lab.max() + 1
    sizes = np.bincount(g)
    rnd = np.random.default_rng(0).permutation(np.repeat(np.arange(len(sizes)), sizes))
    W1, b1, W2, b2 = mlp_np(m.blocks[b])
    ward = fcluster(linkage(W1 / np.linalg.norm(W1, axis=1, keepdims=True), "ward"), len(sizes), "maxclust") - 1
    U = stream_inputs(m, cal, b).reshape(-1, 128).astype(np.float64)
    Hh = gelu_np(U @ W1.T + b1); Y = Hh @ W2.T + b2
    energy = (Hh ** 2).mean(0) * (W2 ** 2).sum(0)
    arms = {}
    keep = np.sort(np.argsort(-energy)[:len(sizes)]); arms["cut_ls"] = (W1[keep], b1[keep])
    for gname, gg in (("family", g), ("random", rnd), ("ward", ward)):
        ks = np.unique(gg)
        arms[f"{gname}_merge_ls"] = (np.stack([W1[gg == k].mean(0) for k in ks]), np.array([b1[gg == k].mean() for k in ks]))
        rep = np.array([np.where(gg == k)[0][np.argmax(energy[gg == k])] for k in ks])
        arms[f"{gname}_rep_ls"] = (W1[rep], b1[rep])
    out = {"K": int(len(sizes)), "family_count": int((n_s > 0).sum()), "group_size_max": int(sizes.max()), "arms": {}}
    cor = {}
    base = load_net("L", 128, 512, 25)
    a0 = acc(base, d, sealed); out["intact"] = {"fine": a0["fine"], "super": a0["super"]}
    for name, (W1n, b1n) in arms.items():
        W2n, b2n = ls_out(gelu_np(U @ W1n.T + b1n), Y)
        mm = load_net("L", 128, 512, 25); set_mlp(mm.blocks[b], W1n, b1n, W2n, b2n)
        a = acc(mm, d, sealed); cor[name] = a; out["arms"][name] = {"fine": a["fine"], "super": a["super"]}
        print(b, name, out["arms"][name], flush=True)
    out["vs_cut"] = {n: {k: boot_diff(cor[n][f"{k}_correct"], cor["cut_ls"][f"{k}_correct"]) for k in ("fine", "super")} for n in cor}
    out["family_vs_random"] = {k: boot_diff(cor["family_merge_ls"][f"{k}_correct"], cor["random_merge_ls"][f"{k}_correct"]) for k in ("fine", "super")}
    v, r = out["vs_cut"]["family_merge_ls"], out["family_vs_random"]
    out["pass"] = bool(v["fine"][1] > 0 and v["super"][1] > 0 and r["fine"][1] > 0 and r["super"][1] > 0)
    print(b, "vs cut", out["vs_cut"], "pass", out["pass"], flush=True)
    res[f"b{b}"] = out
save("v3", res)
