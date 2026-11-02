"""Pictures for N11: held-out images the two-block cut damaged most; intact / cut / closed form / trained answers."""
import sys
from PIL import Image, ImageDraw
src = open("n11_multi.py").read().split("KCUT = float(")[0]
sys.argv = ["x", "gn", "17,16", "0"]; exec(src)
names = [n.split(",")[0] for n in class_names()]
for cb in CBS:
    W2 = np.array(m.blocks[cb].fc2.weight); b2 = np.array(m.blocks[cb].fc2.bias); b2 = b2 + W2[:, REM[cb]] @ MU[cb][REM[cb]]; W2[:, REM[cb]] = 0; m.blocks[cb].fc2 = lin(W2, b2)
LC = np.array(logits_full(EV)); arms = {"I": np.array(LEV), "C": LC}
for tag in ("gn", "tr"):
    z = np.load(os.path.join(HF, "imagenet", f"n11_{tag}_17_16.npz")); keep = [(b, m.blocks[b].fc1, m.blocks[b].fc2) for b in range(15, NB)]
    for b in range(15, NB): m.blocks[b].fc1 = lin(z[f"W1_{b}"], z[f"b1_{b}"]); m.blocks[b].fc2 = lin(z[f"W2_{b}"], z[f"b2_{b}"])
    arms["F" if tag == "gn" else "L"] = np.array(logits_full(EV))
    for b, f1, f2 in keep: m.blocks[b].fc1 = f1; m.blocks[b].fc2 = f2
kl = lambda A: klv(mx.array(A), LEV)
kc, kf, kt = kl(LC), kl(arms["F"]), kl(arms["L"]); worst = np.argsort(-kc)[:16]; T = 110
img = Image.new("RGB", (T * 8 + 16, (T + 64) * 2 + 40), "white"); dr = ImageDraw.Draw(img)
dr.text((8, 6), "Blocks 16+17 at 50% removed. 16 held-out images the cut hurt most. I intact / C cut / F closed form / L trained; KL cut -> F, L", fill="black")
for q, i in enumerate(worst):
    x0, y0 = 8 + (q % 8) * T, 30 + (q // 8) * (T + 64); img.paste(Image.fromarray(np.asarray(X[EV[i]])).resize((T - 6, T - 6)), (x0, y0))
    for r_, tag in enumerate("ICFL"):
        lg = arms[tag][i]; c_ = lg.argmax(); pr = np.exp(lg - lg.max()); pr /= pr.sum()
        dr.text((x0, y0 + T - 4 + 11 * r_), f"{tag} {names[c_][:10]} {pr[c_]:.2f}", fill="darkred" if tag == "C" else "darkgreen")
    dr.text((x0, y0 + T + 40), f"{kc[i]:.2f}->{kf[i]:.2f},{kt[i]:.2f}", fill="black")
fp = os.path.join(HERE, "figs", "l1", "n11_two_blocks.png"); img.save(fp)
top1 = {t: float((arms[t].argmax(1) == y[EV]).mean()) for t in arms}; agree = {t: float((arms[t].argmax(1) == arms["I"].argmax(1)).mean()) for t in arms}
print(fp, "top-1", top1, "agree with intact", agree, "worst-16 KL cut %.3f F %.3f L %.3f" % (kc[worst].mean(), kf[worst].mean(), kt[worst].mean()))
