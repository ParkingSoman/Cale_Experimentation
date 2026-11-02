"""Gate-correlation test: true gate patterns, but each layer's pattern from an independent random image ('layer'), or
4a-4e from one image and 5a-5b from another ('stage'); uniform output metric I - 11^T/n."""
import numpy as np, torch, sys, time
src = open("c20_depth.py").read().split("# pass 1: rates")[0]; exec(src)
mode = sys.argv[1] if len(sys.argv) > 1 else "layer"
def K_shuffled(mode, imgs=TR, rep=10):
    K = torch.zeros(3920, 3920, device=DEV); rng = np.random.default_rng(22)
    for i in range(0, 2000, 10):
        recs = []
        for t in range(3):                                                  # three independent image batches
            bb = np.sort(rng.choice(imgs, 10, replace=False)); x, p = family_inputs(bb); recs.append(record(x, p)[0])
        names = list(recs[0].keys()); rec = {}
        for j, nm in enumerate(names):
            if mode == "layer": rec[nm] = recs[0][nm][torch.from_numpy(rng.permutation(10)).to(DEV)] if False else recs[j % 3][nm][torch.from_numpy(rng.permutation(10)).to(DEV)]
            else: rec[nm] = recs[0][nm] if (nm.startswith("4a") or "mixed4" in nm or nm == "pool4e") else recs[1][nm]
        xi = (lambda z: z - z.mean(1, keepdim=True))(torch.randn(10 * rep, 1008, device=DEV))
        G, P = lin_true(rec, rep); g = fam_grads(G, P, xi); K += g.T @ g
    return (K / 2000).cpu().double()
K = K_shuffled(mode); torch.save(K.float(), f"cache/c22_K_{mode}.pt")
e = torch.linalg.eigvalsh(K).clamp(min=0); pe = e / e.sum(); print(f"[{mode}] effective rank {float(torch.exp(-(pe[pe > 0] * pe[pe > 0].log()).sum())):.0f}", flush=True)
refs = REFS(); refs["4d"] = np.load("cache/c8_ladder_4b_4d_5b_logits.npz")["4d"]; refs["true_a3"] = np.load("cache/c20_depth_a1_a2_a3_b_b_uni.npz")["a3"]
r = factor_build(K, f"c22_build_{mode}"); paired(mode, r, refs); np.save(f"cache/c22_{mode}.npy", r)
