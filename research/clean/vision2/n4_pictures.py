"""Pictures for the closed-form nudge: read side (crops) and write side (repaired predictions) on held-out images."""
import sys
from n_harness import *
from PIL import Image, ImageDraw
names = [n.split(",")[0] for n in class_names()]; T = 88; k = 8; CROP = {2: 136, 3: 224}
def crop(i, p, g):
    st = {14: 2, 7: 3}[g]; r, c = divmod(int(p), g); s = 224 / g; cs = CROP[st]; cy, cx = (r + .5) * s, (c + .5) * s
    t = int(np.clip(cy - cs / 2, 0, 224 - cs)); l = int(np.clip(cx - cs / 2, 0, 224 - cs)); return Image.fromarray(np.asarray(X[i])[t:t + cs, l:l + cs]).resize((T, T))
for key in sys.argv[1:]:
    f = Fam(key); z = np.load(os.path.join(HERE, "figs", "l1", f"cf_fisher_{key}.npz")); CF = {"u": z["u"], "b0": float(z["b0"]), "w": z["w"]}; dB = z["dbias"].astype(np.float32)
    pre = np.concatenate([np.array(m.blocks[f.pb].pre(f.XE[a:b])) for a, b in batches(f.XE.shape[0])]); n, g = pre.shape[0], pre.shape[1]; pre = pre.reshape(n, g * g, -1)
    gl = lambda zz: zz * 0.5 * (1 + np.tanh(0.7978845608 * (zz + 0.044715 * zz ** 3)))
    hp = gl(pre @ f.parent["u"] + f.parent["b0"]); hL = gl(pre @ f.L["u"] + f.L["b0"]); hC = gl(pre @ CF["u"] + CF["b0"])
    d = np.load(os.path.join(HERE, "figs", "l1", f"acts_{key}.npz")); hc = d["hc"][:, 0].reshape(n, -1)
    sp = lambda sc: [(i, sc[i].argmax()) for i in np.argsort(-sc.max(1))[:k]]
    rows = [("parent", sp(hp), hp), ("child (hand-studied)", sp(hc), hc), ("trained neuron (L1)", sp(hL), hL), ("CLOSED-FORM neuron", sp(hC), hC),
            ("on the child's top spots: parent", sp(hc), hp), ("   same spots: trained", sp(hc), hL), ("   same spots: closed form", sp(hc), hC)]
    # write side: images the cut hurt most
    lgI = np.array(f.TE); lgC = np.array(f.run(f.XE, cut=True)); lgF = np.array(f.run(f.XE, CF, dB, cut=True)); lgL = np.array(f.run(f.XE, f.L, cut=True))
    kc = f.k_naive; kF = f.evalkl(CF, dB); kL = f.evalkl(f.L); worst = np.argsort(-kc)[:k]
    W = T * k + 16; img = Image.new("RGB", (W, (T + 30) * (len(rows) + 1) + 120), "white"); dr = ImageDraw.Draw(img)
    dr.text((8, 6), f"{key}: read side (crops, % of each row's max) and write side (bottom: held-out images the child cut hurt most)", fill="black"); yy = 34
    for cap, spots, val in rows:
        dr.text((8, yy - 12), cap, fill="black"); mv = max(val.max(), 1e-6)
        for j, (i, q) in enumerate(spots): img.paste(crop(f.ev[i], q, g), (8 + j * T, yy)); dr.text((8 + j * T, yy + T), f"{names[y[f.ev[i]]][:9]} {100 * val[i, q] / mv:.0f}%", fill="darkgreen")
        yy += T + 30
    dr.text((8, yy - 12), "worst images for the cut: intact answer / cut / closed form / trained, and KL of cut -> closed form", fill="black")
    for j, i in enumerate(worst):
        img.paste(Image.fromarray(np.asarray(X[f.ev[i]])).resize((T, T)), (8 + j * T, yy))
        for r_, (lg, tag) in enumerate(((lgI, "I"), (lgC, "C"), (lgF, "F"), (lgL, "L"))):
            c_ = lg[i].argmax(); pr = np.exp(lg[i] - lg[i].max()); pr /= pr.sum()
            dr.text((8 + j * T, yy + T + 2 + 11 * r_), f"{tag} {names[c_][:8]} {pr[c_]:.2f}", fill="darkred" if tag == "C" else "darkgreen")
        dr.text((8 + j * T, yy + T + 46), f"{kc[i]:.2f}->{kF[i]:.2f}", fill="black")
    fp = os.path.join(HERE, "figs", "l1", f"closed_{key}.png"); img.save(fp); print(fp, "worst-8 KL: cut %.3f closed %.3f trained %.3f" % (kc[worst].mean(), kF[worst].mean(), kL[worst].mean()))
