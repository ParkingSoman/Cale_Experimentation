"""D1f: the consensus form across MANY small neurons. Random small-model neurons in one stage; for each: top-10 tiny parts
(tiny stages s-1 pooled and s), then (description, no fitting): corr of sum-of-GELUs, max, best part, and
GELU(S - theta) with S = sum of signed z-scored votes, theta on a grid; theta* = best theta. Candidate "constants":
theta*/std(S); rate match: P(S > theta*) vs the parts' mean firing rate and vs the small neuron's own rate.
usage: python d1_consensus.py SMALL STAGE(0-based) N [K]"""
import os, sys, json
import numpy as np, mlx.core as mx
from scipy.special import erf
from convnext import *
X, y = load_images(); gl = lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2)))
CFG = {"tiny": ((3, 3, 9, 3), (96, 192, 384, 768)), "atto": ((2, 2, 6, 2), (40, 80, 160, 320)), "nano": ((2, 2, 8, 2), (80, 160, 320, 640))}
small, stage, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); K = int(sys.argv[4]) if len(sys.argv) > 4 else 10
M = {k: load_convnext(k, *CFG[k]) for k in ("tiny", small)}; ana, _ = split(y); I = ana[::25]; rng = np.random.default_rng(1); R = 56 // 2 ** stage
pick = np.stack([rng.choice(R * R, 25, replace=False) for _ in I])
def collect(name, stages):
    m = M[name]; ST = m.stage_of; out, ids = [], []
    for b in range(len(m.blocks)):
        if ST[b] in stages: ids += [(b, n) for n in range(m.blocks[b].fc1.weight.shape[0])]
    for a in range(0, len(I), 50):
        x = m.stem_norm(m.stem(mx.array(norm_img(np.asarray(X[I[a:a + 50]]))))); acc = []; B = x.shape[0]
        idx = mx.array((np.arange(B)[:, None] * R * R + pick[a:a + B]).reshape(-1))
        for b in range(len(m.blocks)):
            if b > 0 and ST[b] != ST[b - 1]: x = m.down[ST[b] - 1](m.down_norm[ST[b] - 1](x))
            if ST[b] in stages:
                z = m.blocks[b].fc1(m.blocks[b].pre(x))
                if ST[b] < stage: Bz, Rz, _, Kz = z.shape; z = z.reshape(Bz, Rz // 2, 2, Rz // 2, 2, Kz).mean((2, 4))
                acc.append(z.reshape(-1, z.shape[-1])[idx])
            if ST[b] > max(stages): break
            x = m.blocks[b](x)
        out.append(np.array(mx.concatenate(acc, 1)).astype(np.float32))
    return np.concatenate(out), ids
Zs, ids_s = collect(small, [stage]); Zt, ids_t = collect("tiny", [stage - 1, stage]) if stage > 0 else collect("tiny", [stage])
Ht = gl(Zt); Htz = (Ht - Ht.mean(0)) / (Ht.std(0) + 1e-9); del Ht
Ztz = (Zt - Zt.mean(0)) / (Zt.std(0) + 1e-9)
# sample small neurons that are "alive" (fire on >= 0.5% of positions)
alive = np.where((Zs > 0).mean(0) > 0.005)[0]; S_ix = rng.choice(alive, N, replace=False); TH = np.arange(0, 33, 2)
rows = []
for j in S_ix:
    t = gl(Zs[:, j].astype(np.float64)); tz = (t - t.mean()) / t.std(); c = (Htz * tz[:, None].astype(np.float32)).mean(0)
    P = np.argsort(-c)[:K]; Hp = gl(Zt[:, P].astype(np.float64)); hs = Hp / (Hp.std(0) + 1e-9)
    sg = np.sign([np.corrcoef(Zt[:, p], t)[0, 1] for p in P]); S = (Ztz[:, P] * sg).sum(1).astype(np.float64)
    cc = lambda v: float(np.corrcoef(v, t)[0, 1]) if v.std() > 0 else 0.0
    ct = [cc(gl(S - th)) for th in TH]; k = int(np.argmax(ct)); th = TH[k]
    on = Ztz[:, P] > 2; one = on.sum(1) == 1; many = on.sum(1) >= 3
    rows.append(dict(id=ids_s[j], best_part_corr=float(c[P[0]]), sum=cc(hs.sum(1)), max=cc(hs.max(1)), single=cc(hs[:, 0]), cons=ct[k], theta=float(th),
                     theta_over_sd=float(th / S.std()), rate_S=float((S > th).mean()), rate_parts=float((Zt[:, P] > 0).mean()), rate_small=float((Zs[:, j] > 0).mean()),
                     one=float(tz[one].mean()) if one.sum() > 20 else np.nan, many=float(tz[many].mean()) if many.sum() > 20 else np.nan,
                     curve=[round(v, 3) for v in ct]))
print(f"{small} stage {stage}: {N} random alive neurons, top-{K} tiny parts, {len(I) * 25} positions")
print(f"{'small':10s} {'best':>5s} {'sum':>5s} {'max':>5s} {'cons':>5s} {'th*':>4s} {'th/sd':>6s} {'P(S>th)':>8s} {'r_parts':>8s} {'r_small':>8s} {'1 on':>6s} {'3+ on':>6s}")
for r in sorted(rows, key=lambda r: -r["best_part_corr"]):
    print(f"{str(r['id']):10s} {r['best_part_corr']:5.2f} {r['sum']:5.2f} {r['max']:5.2f} {r['cons']:5.2f} {r['theta']:4.0f} {r['theta_over_sd']:6.2f} {r['rate_S']:8.4f} {r['rate_parts']:8.4f} {r['rate_small']:8.4f} {r['one']:6.2f} {r['many']:6.2f}")
A = lambda k: np.array([r[k] for r in rows], float)
good = A("best_part_corr") > .3
print("medians (all / best part corr > .3, n=%d): " % good.sum() + "  ".join(f"{k} {np.nanmedian(A(k)):.3f}/{np.nanmedian(A(k)[good]):.3f}" for k in ("sum", "max", "single", "cons", "theta_over_sd", "one", "many")))
print("corr across neurons: log P(S>th*) vs log r_small %.2f; vs log r_parts %.2f; th/sd vs log r_small %.2f" % (
    np.corrcoef(np.log(A("rate_S") + 1e-5), np.log(A("rate_small") + 1e-5))[0, 1], np.corrcoef(np.log(A("rate_S") + 1e-5), np.log(A("rate_parts") + 1e-5))[0, 1],
    np.corrcoef(A("theta_over_sd"), np.log(A("rate_small") + 1e-5))[0, 1]))
json.dump(rows, open(os.path.join(HF, "imagenet", "d1", f"consensus_{small}_s{stage}.json"), "w"), default=str)
