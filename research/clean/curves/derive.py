"""Derived coarse curve neuron (no training, no knob fitted to the loss). See CURVES-LOG 'Derivation'.
usage: derive.py [r] [--intercept 0/1] [--eval dev|sealed]"""
import argparse, json, numpy as np, torch
from inception import *
from family import Family, Data, evaluate, pool3, IN4A
PREF = np.deg2rad(np.array([10, 50, 70, 120, 170, 180, 250, 290, 320, 350]))   # orientation order (S1)

def derived_read(m, fam, r, ks=13):
    """K[:, a, b] = sum_theta W_theta[:, a - dy, b - dx] (centred), where member theta is read at q - r u_theta."""
    W = m.mixed3b_5x5_pre_relu_conv.weight.detach().cpu().numpy()[[n - 320 for n in fam]]
    b = m.mixed3b_5x5_pre_relu_conv.bias.detach().cpu().numpy()[[n - 320 for n in fam]]
    K = np.zeros((32, ks, ks), np.float32); c = ks // 2
    for i, t in enumerate(PREF):
        dy, dx = int(round(r * np.sin(t))), int(round(-r * np.cos(t)))    # member read at q + (dy, dx)
        K[:, c + dy - 2:c + dy + 3, c + dx - 2:c + dx + 3] += W[i]
    return K, float(b.sum())

def bayes_write(m, fm, fam, D, n=1000, seed=11, intercept=True):
    """Per 4a input conv: column = sum_theta beta_theta col_theta, beta from pooled member vs pooled coarse."""
    idx = np.sort(np.random.default_rng(seed).choice(D.tr, n, replace=False))
    S = {k: 0.0 for k in ["c", "cc", "n"]}; Sm = {"x": np.zeros(10), "xc": np.zeros(10)}
    Sp = {"c": 0.0, "cc": 0.0, "n": 0.0}; Spm = {"x": np.zeros(10), "xc": np.zeros(10)}
    with torch.no_grad():
        for i in range(0, n, 100):
            ins = inputs_3b(m, to_3a(m, prep(D.X[idx[i:i+100]])))
            z = F.relu(pre_3b(m, ins[0])[:, fam]); c = F.relu(fm.pre(ins))
            for (zz, cc, A, B) in [(pool3(z), pool3(c), S, Sm),
                                   (F.max_pool2d(F.pad(pool3(z), (1, 1, 1, 1), value=-1e9), 3, 1), F.max_pool2d(F.pad(pool3(c), (1, 1, 1, 1), value=-1e9), 3, 1), Sp, Spm)]:
                x = zz.permute(0, 2, 3, 1).reshape(-1, 10).cpu().numpy().astype(np.float64); cv = cc.reshape(-1).cpu().numpy().astype(np.float64)
                A["c"] += cv.sum(); A["cc"] += (cv * cv).sum(); A["n"] += len(cv); B["x"] += x.sum(0); B["xc"] += (x * cv[:, None]).sum(0)
    def fit(A, B):
        mc = A["c"] / A["n"]; vc = A["cc"] / A["n"] - mc**2; mx = B["x"] / A["n"]
        beta = (B["xc"] / A["n"] - mx * mc) / vc; a = mx - beta * mc
        return beta, (a if intercept else np.zeros(10))
    (b1, a1), (b2, a2) = fit(S, Sm), fit(Sp, Spm)
    cols, cbias = [], []
    for j, name in enumerate(IN4A):
        C0 = getattr(m, name).weight.detach().cpu()[:, fam, 0, 0].double().numpy()
        beta, a = (b2, a2) if j == 3 else (b1, a1)
        cols.append(torch.tensor(C0 @ beta, dtype=torch.float32)[:, None]); cbias.append(torch.tensor(C0 @ a, dtype=torch.float32).to(DEV))
    return cols, cbias, b1, a1

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("r", type=float, nargs="?", default=4)
    ap.add_argument("--intercept", type=int, default=1); ap.add_argument("--eval", default="dev"); ap.add_argument("--ks", type=int, default=13)
    ap.add_argument("--dial", type=int, default=0)
    a = ap.parse_args()
    m = load_model(); D = Data(seed=0, ndev=1000 if a.eval == "dev" else 0)
    fam = list(np.load("cache/s1_fine.npz")["fam"])
    fm = Family(m, fam, 1, ks=a.ks).to(DEV)
    K, b = derived_read(m, fam, a.r, ks=a.ks)
    _, _, cols0 = fm.weights()
    fm.set_weights(torch.tensor(K)[None].to(DEV), torch.tensor([b]).to(DEV), [c.to(DEV) for c in cols0])
    if a.dial:   # early layers are dials (essay): bias so the ReLU never clips on natural images
        with torch.no_grad():
            ins = inputs_3b(m, to_3a(m, prep(D.X[np.sort(D.tr[:200])]))); z = fm.pre(ins)
            b = b - float(z.min())
        fm.set_weights(torch.tensor(K)[None].to(DEV), torch.tensor([b]).to(DEV), [c.to(DEV) for c in cols0])
    cols, cbias, beta, alpha = bayes_write(m, fm, fam, D, intercept=bool(a.intercept))
    fm.set_weights(torch.tensor(K)[None].to(DEV), torch.tensor([b]).to(DEV), [c.to(DEV) for c in cols]); fm.colbias = cbias
    idx = D.dv if a.eval == "dev" else D.se
    kl_, se_, top = evaluate(m, fm, D, idx)
    with torch.no_grad():
        ins = inputs_3b(m, to_3a(m, prep(D.X[np.sort(D.tr[:200])]))); rate = (fm.pre(ins) > 0).float().mean().item()
    np.set_printoptions(precision=3, suppress=True)
    print(f"derived r={a.r} intercept={a.intercept} dial={a.dial} ks={a.ks}: {a.eval} KL {kl_:.4f} (se {se_:.4f}) top-1 {top:.4f} | firing rate {rate:.3f} | beta {beta} alpha/100 {alpha/100}")
    torch.save(dict(state=fm.state_dict(), colbias=[c.cpu() for c in cbias], fam=fam, k=1, ks=a.ks, r=a.r), f"cache/ckpt/derived-r{a.r:g}-i{a.intercept}-d{a.dial}.pt")
    json.dump(dict(r=a.r, intercept=a.intercept, split=a.eval, kl=kl_, se=se_, top1=top, rate=rate, beta=beta.tolist()),
              open(f"../../../results/curves-derived-r{a.r:g}-i{a.intercept}-d{a.dial}-{a.eval}.json", "w"), indent=1)
