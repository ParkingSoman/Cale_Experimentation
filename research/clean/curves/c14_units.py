"""I3: which 4d channels the level-0 error reaches through the weights-only composed map, and what they are."""
import numpy as np, torch, sys
from PIL import Image
exec(open("c8_ladder.py").read().split("# sanity: linearity")[0])                 # forward_lin, mlin, circ tools
import fv
C = torch.load("cache/x2_b_level0_p.pt", map_location="cpu")
fm0 = b3_train.FamilyRA(m, fam, 1); fm0.add_rb(m); fm0.add_ra(m); fm0.load_state_dict(C); cs = [float(c) for c in fm0.cscale]
Y = torch.cat([torch.cat([fm0.dW[j].detach() * cs[j], fm0.cbp[j].detach()[:, None] * 0.1], 1) for j in range(4)]).double()
Wc = torch.cat([w.cpu() for w in WF]).double(); nx = sum(w.shape[0] for w in WF[:3])
Lams = [torch.linalg.lstsq(Wc[:nx], Y[:nx]).solution.float().to(DEV), torch.linalg.lstsq(Wc[nx:], Y[nx:]).solution.float().to(DEV)]
ix = np.sort(np.random.default_rng(3).choice(D.dv, 300, replace=False)); en = torch.zeros(528, device=DEV)
with torch.no_grad():
    for i in range(0, 300, 20):
        x, p = family_inputs(ix[i:i + 20]); rs = []
        for t, L in zip((x, p), Lams):
            z = torch.cat([t[:, keep], torch.ones_like(t[:, :1])], 1); rs.append(t[:, fam] - torch.einsum("fa,bayx->bfyx", L, z))
        o = forward_lin(torch.stack(rs, 1), "4d"); en += o.pow(2).sum((0, 2, 3))
share = (en / en.sum()).cpu().numpy(); top = np.argsort(-share)[:12]
print("top-12 4d channels", top.tolist(), "shares", np.round(share[top], 3).tolist(), f"total {share[top].sum():.3f}", flush=True)
def pre4d(h3b):
    h = m.maxpool4(F.pad(h3b, (0, 1, 0, 1), value=float('-inf')), kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False)
    for n in ["mixed4a", "mixed4b", "mixed4c"]: h = mixed(m, n, h)
    g = lambda s: getattr(m, f"mixed4d_{s}")
    a = g("1x1_pre_relu_conv")(h); b = g("3x3_pre_relu_conv")(F.pad(g("3x3_bottleneck")(g("3x3_bottleneck_pre_relu_conv")(h)), (1, 1, 1, 1)))
    c = g("5x5_pre_relu_conv")(F.pad(g("5x5_bottleneck")(g("5x5_bottleneck_pre_relu_conv")(h)), (2, 2, 2, 2)))
    d = g("pool_reduce_pre_relu_conv")(g("pool")(F.pad(h, (1, 1, 1, 1), value=float('-inf')), kernel_size=(3, 3), stride=(1, 1), padding=0, ceil_mode=False))
    return torch.cat([a, b, c, d], 1)
ims = np.sort(np.random.default_rng(9).choice(D.tr, 2000, replace=False)); mx, mn = [], []
with torch.no_grad():
    for i in range(0, 2000, 50):
        z = pre4d(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(D.X[ims[i:i + 50]])))[0])))[:, top].flatten(2)
        mx.append(z.amax(2)); mn.append(z.amin(2))
mx, mn = torch.cat(mx), torch.cat(mn)
def crop(img_i, pos, size=96):
    a = np.asarray(D.X[img_i]); H = a.shape[0]; y, x = divmod(int(pos), 14); cy, cx = int((y + .5) * H / 14), int((x + .5) * H / 14)
    y0, x0 = np.clip(cy - size // 2, 0, H - size), np.clip(cx - size // 2, 0, H - size); return np.asarray(Image.fromarray(a[y0:y0 + size, x0:x0 + size].astype(np.uint8)).resize((128, 128)))
mr = load_model(redirected=True); rows = []
for k, u in enumerate(top):
    vis = fv.visualize(mr, lambda ins: pre4d(F.relu(pre_3b(mr, ins[0]))), int(u))
    tiles = [vis]
    with torch.no_grad():
        for sel in [torch.argsort(-mx[:, k])[:4], torch.argsort(mn[:, k])[:2]]:
            for j in sel.tolist():
                z = pre4d(F.relu(pre_3b(m, inputs_3b(m, to_3a(m, prep(D.X[ims[j:j + 1]])))[0])))[0, u].flatten()
                pos = int(z.argmax()) if len(tiles) < 5 else int(z.argmin()); tiles.append(crop(ims[j], pos))
    rows.append(np.concatenate(tiles, 1)); print(f"unit {u} done", flush=True)
Image.fromarray(np.concatenate(rows, 0)).save("cache/c14_units.png"); print("saved cache/c14_units.png (columns: feature vis | 4 top crops | 2 most negative crops)")
