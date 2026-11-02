"""R3b (NEURON-MATH Prop. 11.1): is the trained reader change Delta-W (and B) inside span(W_F)? Weights only."""
import sys, numpy as np, torch
from inception import *
from family import IN4A
m = load_model(); fam = list(np.load("cache/s1_fine.npz")["fam"])
WF = torch.cat([getattr(m, n).weight.detach()[:, fam, 0, 0] for n in IN4A], 0).cpu()          # (368, 10)
Q, _ = torch.linalg.qr(WF)
rng = torch.Generator().manual_seed(0)
for tag in sys.argv[1:]:
    st = torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu", weights_only=False)["state"]
    from family import Family
    import b3_train
    fm = b3_train.FamilyRA(m, fam, torch.load(f"cache/ckpt/{tag}.pt", map_location="cpu", weights_only=False)["k"]); fm.add_rb(m); fm.add_ra(m); fm.load_state_dict(st)
    dW = torch.cat([d.detach() * s for d, s in zip(fm.dW, fm.cscale)], 0)                     # (368, 470) raw units
    B = torch.cat([c.detach() * s for c, s in zip(fm.cols, fm.cscale)], 0)
    ins = lambda X: float((Q.T @ X).pow(2).sum() / X.pow(2).sum())
    R, _ = torch.linalg.qr(torch.randn(368, 10, generator=rng))
    print(f"{tag}: dW energy in span(W_F) {ins(dW):.3f} (random 10-dim {float((R.T @ dW).pow(2).sum() / dW.pow(2).sum()):.3f}); "
          f"B in span {ins(B):.3f}; |dW| {float(dW.norm()):.3f} vs |W_rest| {float(torch.cat([getattr(m, n).weight.detach()[:, fm.keep, 0, 0] for n in IN4A]).norm()):.3f}")
    S = torch.linalg.svdvals(dW); print(f"   top singular values share (1, 3, 10): {float(S[:1].pow(2).sum() / S.pow(2).sum()):.2f} {float(S[:3].pow(2).sum() / S.pow(2).sum()):.2f} {float(S[:10].pow(2).sum() / S.pow(2).sum()):.2f}")
