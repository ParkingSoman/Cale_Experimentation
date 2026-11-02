"""T9: is the drive LOCAL? For each neuron's top-40 images (10k set): cut a 112 px window around the peak and paste
it on mid-gray (a) at the SAME frame position, (b) at the CENTRE. Then edits of the centred patch: flip, vflip,
rot90, rot180, gray, shuffle 4x4 inside the patch. Value = max over the map / original peak (median).
PREDICTION (11 n1031, before run): the drive is local hanging-fur texture with a handedness: the centred patch keeps
>= 50%; vflip and rot90 of the patch drop it below 25%; left-right flip drops it to <= 60%."""
from mid_common import *
G._init(); ana = np.load(os.path.join(OUTD, "t2_ana10.npy")); rng = np.random.default_rng(0)
T = [(7, 604), (8, 4), (10, 1316), (11, 1031)]; W = 112


def place(p, r0, c0, bg=124):
    im = np.full((224, 224, 3), bg, np.uint8); im[r0:r0 + W, c0:c0 + W] = p; return im


def shuf(p, k=4):
    s = W // k; t = p.reshape(k, s, k, s, 3).transpose(0, 2, 1, 3, 4).reshape(k * k, s, s, 3)[rng.permutation(k * k)]
    return t.reshape(k, k, s, s, 3).transpose(0, 2, 1, 3, 4).reshape(W, W, 3)


for t in T:
    z = np.load(os.path.join(OUTD, f"t2_z_{t[0]}_{t[1]}.npy")).astype(np.float32); f = z.reshape(len(z), -1)
    top = np.argsort(-f.max(1))[:40]; base = f[top].max(1); imgs = np.asarray(G._X[ana[top]])
    pr, pc = np.divmod(f[top].argmax(1), 14); r0 = np.clip(((pr + .5) * 16 - W / 2).astype(int), 0, 224 - W); c0 = np.clip(((pc + .5) * 16 - W / 2).astype(int), 0, 224 - W)
    P = [imgs[j, r0[j]:r0[j] + W, c0[j]:c0[j] + W] for j in range(40)]; mid = (224 - W) // 2
    sets = {"same place": [place(p, a, b) for p, a, b in zip(P, r0, c0)], "centre": [place(p, mid, mid) for p in P],
            "c flip": [place(p[:, ::-1], mid, mid) for p in P], "c vflip": [place(p[::-1], mid, mid) for p in P],
            "c rot90": [place(np.rot90(p), mid, mid) for p in P], "c rot180": [place(p[::-1, ::-1], mid, mid) for p in P],
            "c gray": [place(np.repeat(p.mean(2, keepdims=True), 3, 2).astype(np.uint8), mid, mid) for p in P], "c shuf4": [place(shuf(p), mid, mid) for p in P],
            "same, own-mean bg": [place(p, a, b, bg=int(im.mean())) for p, a, b, im in zip(P, r0, c0, imgs)],
            "outside only": [np.where(np.pad(np.ones((W, W, 1), bool), ((a, 224 - W - a), (b, 224 - W - b), (0, 0))), 124, im).astype(np.uint8) for a, b, im in zip(r0, c0, imgs)]}
    out = {k: np.median(zmaps(np.stack(v), [t])[t].reshape(40, -1).max(1) / base) for k, v in sets.items()}
    print(t, " ".join(f"{k}: {v:.2f}" for k, v in out.items()), flush=True)
