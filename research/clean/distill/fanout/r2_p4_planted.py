"""P4 round 2: the MIXTURE PLANTED TEST (the test that can fail). CLIP MLP-only.

Plant: T' = T with every CUT MLP unit (floor ranking on X, at fraction f) made a copy of one random KEPT unit of the
same block (fc row and bias), and optionally a mixture: fc_c = (1-eps) fc_j + eps fc_rand (eps = 0 exact copy).
The cut units' message is then (almost) exactly predictable from kept firings. Prediction (written before the run):
  fold KL(T' -> fold) < 5% of KL(T' -> slice) at eps = 0; the mean null recovers much less (> 50% of slice KL).
  If the fold does NOT recover the planted copies, the fold tool is broken (not the network).
Scoring: KL on a FRESH pink draw (seed 7, n = 128), against T' logits (teacher class embeddings, scale 100). No photos.
"""
import os
import sys

import torch
import torch.nn.functional as F

FAN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FAN)
sys.path.insert(0, os.path.dirname(FAN))
sys.path.insert(0, os.path.join(os.path.dirname(FAN), "clipkd"))
import generic  # noqa: E402
import vitgen  # noqa: E402
import eval_fast  # noqa: E402
from rules import noisecut, r2_p4_lib as L, p4write as P  # noqa: E402

DEV = noisecut.DEV


@torch.no_grad()
def logits(W, X, cls, scale):
    Wd = vitgen.to(W, DEV)
    out = []
    for a in range(0, len(X), 32):
        e = vitgen.forward(Wd, X[a:a + 32].to(DEV).float()).float()
        out.append((scale * F.normalize(e, dim=-1) @ cls).cpu())
    return torch.cat(out)


def kl(zr, z):
    lr, ls = F.log_softmax(zr, 1), F.log_softmax(z, 1)
    return float((lr.exp() * (lr - ls)).sum(1).mean()), float((zr.argmax(1) == z.argmax(1)).float().mean())


def main():
    f = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    T = torch.load(os.path.join(FAN, "cache", "clip_teacher_visual.pt"), weights_only=True)
    X = torch.load(os.path.join(FAN, "cache", "input_pink_n256_s1.pt"), weights_only=True).float()
    Xe = generic.pink(128, seed=7)
    ref = eval_fast.teacher_ref("test")
    cls, scale = ref["cls"].to(DEV), ref["scale"]
    sc = noisecut.vit_scores(T, X=X)
    D, H, M = vitgen.sizes(1.0, T)
    _, _, m = vitgen.sizes(f, T)
    mk = [noisecut.top(s, m) for s in sc["mlp"]]
    g = torch.Generator().manual_seed(0)
    Tp = {k: v for k, v in T.items()}
    Tp["blocks"] = []
    for bi, B in enumerate(T["blocks"]):
        B = dict(B)
        fw, fb = B["fc_w"].clone(), B["fc_b"].clone()
        kept = mk[bi]
        cut = torch.ones(M, dtype=torch.bool)
        cut[kept] = False
        cut = cut.nonzero().flatten()
        src = kept[torch.randint(len(kept), (len(cut),), generator=g)]
        rnd = cut[torch.randperm(len(cut), generator=g)]
        fw[cut] = (1 - eps) * fw[src] + eps * B["fc_w"][rnd]
        fb[cut] = (1 - eps) * fb[src] + eps * B["fc_b"][rnd]
        B["fc_w"], B["fc_b"] = fw, fb
        Tp["blocks"].append(B)
    keep = dict(stream=torch.arange(D), heads=[torch.arange(H)] * len(T["blocks"]), mlp=mk)
    zT = logits(Tp, Xe, cls, scale)
    rows = {"slice": vitgen.cut(Tp, keep["stream"], keep["heads"], mk),
            "fold": L.build_vit2(Tp, f, X, sets=("mlp",), keep=keep, tag="planted"),
            "meannull": L.build_vit2(Tp, f, X, sets=("mlp",), keep=keep, null="mean", tag="planted_null")}
    zO = logits(T, Xe, cls, scale)
    print(f"PLANTED f={f} eps={eps}: KL(T'->T) {kl(zT, zO)[0]:.3f} (how far the plant moved the net)")
    for k_, W in rows.items():
        a, b = kl(zT, logits(W, Xe, cls, scale))
        print(f"PLANTED f={f} eps={eps} {k_}: KL {a:.4f} agree {b:.3f}  (fresh pink n=128, T' as teacher)")


if __name__ == "__main__":
    main()
