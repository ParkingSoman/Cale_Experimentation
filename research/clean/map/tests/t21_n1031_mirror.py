"""T21 (11 n1031): is there a mirror twin for the right rim? On 300 images (top-150 of n1031 + 150 random), compare
n1031's map on the ORIGINAL with every block-11 neuron's map on the MIRRORED image (mirrored back). Top correlations."""
from mid_common import *
G._init(); ana = np.load(os.path.join(OUTD, "t2_ana10.npy")); rng = np.random.default_rng(6)
z = np.load(os.path.join(OUTD, "t2_z_11_1031.npy")).astype(np.float32).reshape(-1, 196)
sel = np.r_[np.argsort(-z.max(1))[:150], rng.choice(len(z), 150, replace=False)]; imgs = np.asarray(G._X[ana[np.sort(sel)]])
m = G._m


def all11(ims):
    out = []
    for i in range(0, len(ims), 25):
        x = m.stream(mx.array(norm_img(ims[i:i + 25])), upto=11); out.append(np.array(m.blocks[11].fc1(m.blocks[11].pre(x))))
    return np.concatenate(out)


A = all11(imgs); B = all11(imgs[:, :, ::-1].copy())[:, :, ::-1]   # mirrored back
a = np.maximum(A[..., 1031], 0).reshape(-1); Bf = np.maximum(B, 0).reshape(-1, B.shape[-1])
Af = np.maximum(A, 0).reshape(-1, A.shape[-1])
def corr(u, M): u = u - u.mean(); M = M - M.mean(0); return (M.T @ u) / (np.sqrt((M ** 2).sum(0) * (u ** 2).sum()) + 1e-9)
cm = corr(a, Bf); cs = corr(a, Af)
print("mirror-twin candidates (n1031 orig vs k on mirrored):", [(int(k), round(float(cm[k]), 3)) for k in np.argsort(-cm)[:6]])
print("same-image correlates (n1031 vs k, original):     ", [(int(k), round(float(cs[k]), 3)) for k in np.argsort(-cs)[:6]])
print("self mirror corr (n1031 vs n1031 on mirrored):", round(float(cm[1031]), 3))
