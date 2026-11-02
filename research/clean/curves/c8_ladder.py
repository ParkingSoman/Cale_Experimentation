"""A4 (Part VI): the weights-only circuit metric composed to depth L. Linearised network above the readers (ReLU -> 1,
max-pool -> average pool: the gate-free transport that A2 / build g showed is enough at 4a). K^(L) = A_L^T Sigma_L A_L with
A_L = Jacobian of layer L's outputs w.r.t. the family's 3920 signals; Sigma_L = I (weights only), or at the logits the
uniform class covariance I - 11^T / n. Each K gets the factorised build (circ.factor_build). DERIVED metric (weights only); activity moments MIX-DERIVED."""
import numpy as np, torch, time, sys
from circ import *

t0 = time.time(); LAYERS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["4b", "4d", "5b", "logits"]
def nobias(mod):
    for n, mm in mod.named_modules():
        if isinstance(mm, (torch.nn.Conv2d, torch.nn.Linear)) and mm.bias is not None: mm.bias.data.zero_()
import copy; mlin = copy.deepcopy(m); nobias(mlin)
def forward_lin(s, upto):
    """s: (n, 2, 10, 14, 14) family signals (x, p) -> linearised layer output (biases off, so exactly linear)."""
    x, p = s[:, 0], s[:, 1]
    r = [F.conv2d(inp, WF[j][:, :, None, None]) for j, inp in enumerate([x, x, x, p])]
    hh = torch.cat([r[0], mlin.mixed4a_3x3_pre_relu_conv(F.pad(r[1], (1, 1, 1, 1))), mlin.mixed4a_5x5_pre_relu_conv(F.pad(r[2], (2, 2, 2, 2))), r[3]], 1)
    for n in ["4b", "4c", "4d", "4e"]:
        hh = _mixed(n, hh)
        if upto == n: return hh
    hh = F.avg_pool2d(F.pad(hh, (0, 1, 0, 1)), 3, 2)
    for n in ["5a", "5b"]:
        hh = _mixed(n, hh)
        if upto == n: return hh
    lg = mlin.softmax2_pre_activation_matmul(F.avg_pool2d(hh, 7).reshape(-1, 1024))
    return lg - lg.mean(1, keepdim=True)                                                      # (I - 11^T/n) half
def _mixed(n, hh):
    g = lambda s_: getattr(mlin, f"mixed{n}_{s_}")
    a = g("1x1_pre_relu_conv")(hh); b = g("3x3_bottleneck_pre_relu_conv")(hh); c = g("5x5_bottleneck_pre_relu_conv")(hh)
    d = g("pool_reduce_pre_relu_conv")(F.avg_pool2d(F.pad(hh, (1, 1, 1, 1)), 3, 1))
    b = g("3x3_pre_relu_conv")(F.pad(b, (1, 1, 1, 1))); c = g("5x5_pre_relu_conv")(F.pad(c, (2, 2, 2, 2)))
    return torch.cat([a, b, c, d], 1)
# sanity: linearity
s1, s2 = torch.randn(2, 2, 10, 14, 14, device=DEV).unbind(0)
with torch.no_grad():
    o = forward_lin(torch.stack([s1, s2, s1 + s2]), "4b"); print("linearity check", float((o[2] - o[0] - o[1]).norm() / o[2].norm()), flush=True)

def gram(upto, chunk=196):
    eye = torch.eye(3920, device=DEV); outs = []
    with torch.no_grad():
        for i in range(0, 3920, chunk):
            outs.append(forward_lin(eye[i:i + chunk].reshape(-1, 2, 10, 14, 14), upto).flatten(1).half())
    A = torch.cat(outs)                                                                      # (3920, dim)
    K = torch.zeros(3920, 3920, device=DEV)
    for i in range(0, 3920, 980):
        for j in range(0, 3920, 980): K[i:i + 980, j:j + 980] = A[i:i + 980].float() @ A[j:j + 980].float().T
    return K.cpu().double()

res = {}; refs = REFS()
for L in LAYERS:
    K = gram(L); torch.save(K.float(), f"cache/c8_K_{L}.pt")
    d = torch.einsum("sfyxsfyx->yx", K.reshape(2, 10, 14, 14, 2, 10, 14, 14)); d = d / d.max()
    print(f"[{L}] per-position energy corner/centre {float(d[0, 0]):.2f}, edge-mid/centre {float(d[0, 7]):.2f}  ({time.time() - t0:.0f}s)", flush=True)
    res[L] = factor_build(K, f"c8_build_{L}"); paired(L, res[L], refs)
np.savez(f"cache/c8_ladder_{'_'.join(LAYERS)}.npz", **res); print(f"done ({time.time() - t0:.0f}s)")
