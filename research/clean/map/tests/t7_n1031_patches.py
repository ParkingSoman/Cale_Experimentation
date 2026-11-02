"""T7 (11 n1031). PREDICTION (before run): the neuron marks the LEFT contour of a furry mass (fur to the right of the
boundary). A fur patch (96 px) pasted on mid-gray: its left-edge cells score > 2x its right-edge cells; its interior
is negative; a non-fur texture patch (hay, stone wall, wool fabric, honeycomb) at the same place scores < 50% of fur.
Also: class noun check (silky vs wiry coat breeds) from the 10k maps of T2."""
from mid_common import *
G._init(); X, y = G._X, G._y; ana = np.load(os.path.join(OUTD, "t2_ana10.npy")); names = [s.split(",")[0] for s in class_names()]
t = (11, 1031); z = np.load(os.path.join(OUTD, "t2_z_11_1031.npy")).astype(np.float32); f = z.reshape(len(z), -1)
cy = y[ana]; cmax = np.array([f[cy == c].max(1).mean() for c in range(1000)]); cmin = np.array([f[cy == c].min(1).mean() for c in range(1000)])
print("classes by mean max:", [(names[c], round(cmax[c], 2)) for c in np.argsort(-cmax)[:12]])
print("classes by mean min:", [(names[c], round(cmin[c], 2)) for c in np.argsort(cmin)[:12]])
silky = ["Afghan hound", "silky terrier", "Yorkshire terrier", "Maltese dog", "Shih-Tzu", "Lhasa", "English setter", "Irish setter", "Saluki", "Blenheim spaniel"]
wiry = ["wire-haired fox terrier", "Airedale", "otterhound", "Scotch terrier", "giant schnauzer", "standard schnauzer", "Border terrier", "Irish wolfhound", "Lakeland terrier", "Kerry blue terrier"]
smooth = ["boxer", "Doberman", "Weimaraner", "basenji", "Great Dane", "whippet", "Rottweiler", "Labrador retriever", "pug", "Rhodesian ridgeback"]
for nm, L in (("silky", silky), ("wiry", wiry), ("smooth", smooth)):
    ids = [names.index(s) for s in L if s in names]; print(f"{nm:7s} n={len(ids)} mean max {cmax[ids].mean():.2f}  mean min {cmin[ids].mean():.2f}")
print("all classes mean max %.2f; dog classes (151-268) %.2f; non-dog %.2f" % (cmax.mean(), cmax[151:269].mean(), np.r_[cmax[:151], cmax[269:]].mean()))

# patch test
furc = ["komondor", "Old English sheepdog", "Tibetan mastiff", "Newfoundland", "Great Pyrenees", "kuvasz", "briard", "Bouvier des Flandres", "otterhound", "Afghan hound"]
texc = {"hay": ["hay"], "stone wall": ["stone wall"], "wool": ["wool"], "honeycomb": ["honeycomb"], "rapeseed": ["rapeseed"], "brain coral": ["brain coral"]}
anaA, _ = split(y)


def patches(cls_list, k=20):
    out = []
    for c in cls_list:
        ids = anaA[y[anaA] == names.index(c)][:k // len(cls_list) + 2]
        for i in ids: out.append(np.asarray(X[i])[64:160, 64:160])
    return out[:k]


def paste(p, bg=124):
    im = np.full((224, 224, 3), bg, np.uint8); im[64:160, 64:160] = p; return im


reg = {"left": (slice(4, 10), slice(3, 5)), "right": (slice(4, 10), slice(9, 11)), "top": (slice(3, 5), slice(4, 10)), "bottom": (slice(9, 11), slice(4, 10)), "inside": (slice(5, 9), slice(5, 9))}
sets = {"fur": patches(furc)}; sets.update({k: patches(v) for k, v in texc.items()})
sets["fur rot90"] = [np.rot90(p).copy() for p in sets["fur"]]; sets["fur flip"] = [p[:, ::-1].copy() for p in sets["fur"]]
sets["fur meancolor"] = [np.broadcast_to(p.reshape(-1, 3).mean(0).astype(np.uint8), p.shape).copy() for p in sets["fur"]]
print("\npatch on gray: median over 20 patches of region max (cells)")
print("set".ljust(14) + "".join(r.rjust(8) for r in reg) + "   whole-max")
cells = []
for k, L in sets.items():
    ims = np.stack([paste(p) for p in L]); zz = zmaps(ims, [t])[t]
    print(k.ljust(14) + "".join(f"{np.median(zz[:, a, b].max((1, 2))):8.2f}" for a, b in reg.values()) + f"{np.median(zz.max((1, 2))):10.2f}")
    for j in range(3): cells.append((heat(ims[j], zz[j]), f"{k} {zz[j].max():.1f}"))
print(sheet(cells, "t7_n1031_patches.png", ncol=9, size=110, title="11 n1031: patches on gray"))
