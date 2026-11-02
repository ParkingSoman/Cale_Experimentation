"""Phase 5 cold test: circle family, sealed 10k, one run, per-image KL saved, paired differences."""
import json, numpy as np, torch
from inception import *
from family import Data, IN4A
from p5_drop import fam_stats, patched, evaluate_patched
F_ = [462, 82, 300, 311, 446, 255, 352, 315, 363, 454, 323]
m = load_model(); D = Data(seed=0, ndev=0); st = fam_stats(m, F_, D); out = {}; per = {}
def run(name, pw):
    k, top = evaluate_patched(m, pw, D, D.se); per[name] = k; out[name] = dict(kl=float(k.mean()), se=float(k.std() / np.sqrt(len(k))), top1=top)
    print(f"{name:45s} sealed KL {k.mean():.4f} (se {k.std()/np.sqrt(len(k)):.4f}) top-1 {top:.4f}", flush=True)
run("removed (zero)", patched(m, F_, [], st, "zero")); run("removed + constant", patched(m, F_, [], st, "const"))
for keep, lab in [([446], "k=1 rule 446"), ([363], "k=1 naive 363"), ([82, 323], "k=2 rule 82,323"), ([363, 323], "k=2 naive 363,323"),
                  ([82, 363, 323], "k=3 82,363,323"), ([82, 446, 363, 323], "k=4 rule")]:
    for mode in ["const", "bayes"]: run(f"{lab} + {mode}", patched(m, F_, keep, st, mode))
BR = ["mixed3b_1x1_pre_relu_conv", "mixed3b_3x3_pre_relu_conv", "mixed3b_5x5_pre_relu_conv", "mixed3b_pool_reduce_pre_relu_conv"]
for tag in ["p5-circ-k2-derived", "p5-circ-k2-random"]:
    ck = torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu", weights_only=False)
    orig = {n: (getattr(m, n).weight.data.clone(), getattr(m, n).bias.data.clone()) for n in BR}
    for n in BR: getattr(m, n).weight.data.copy_(ck["state"][n][0]); getattr(m, n).bias.data.copy_(ck["state"][n][1])
    run(f"trained k=2 ({tag})", [(ck["state"][n][0].to(DEV), ck["state"][n][1].to(DEV)) for n in IN4A])
    for n in BR: getattr(m, n).weight.data.copy_(orig[n][0]); getattr(m, n).bias.data.copy_(orig[n][1])
def paired(a, b):
    d = per[a] - per[b]; out[f"paired: {a} MINUS {b}"] = dict(diff=float(d.mean()), se=float(d.std() / np.sqrt(len(d))))
    print(f"PAIRED [{a}] - [{b}] = {d.mean():+.4f} (se {d.std()/np.sqrt(len(d)):.4f})", flush=True)
paired("k=1 rule 446 + bayes", "k=1 naive 363 + bayes"); paired("k=2 rule 82,323 + bayes", "k=2 naive 363,323 + bayes")
paired("k=2 rule 82,323 + bayes", "trained k=2 (p5-circ-k2-random)"); paired("trained k=2 (p5-circ-k2-derived)", "trained k=2 (p5-circ-k2-random)")
paired("trained k=2 (p5-circ-k2-derived)", "k=2 rule 82,323 + bayes")
json.dump(out, open("../../../results/curves-p5-cold.json", "w"), indent=1)
