"""T5 (8 n4 family): same key probes on the sibling candidate 7 n570 (strongest negative input, same edit
fingerprint), 6 n1032 (2nd negative input), parent candidate 6 n205, and readers 11 n676 / 11 n745.
Question (no prediction on numbers): does 7 n570 prefer UNBOUNDED horizontal edges (full-width step), where 8 n4
prefers BOUNDED ones (box ends)? Also: zoomed-out natural images (pasted frame) on all."""
from mid_common import *
from PIL import ImageFilter
S = 224; T = [(6, 205), (7, 570), (6, 1032), (8, 4), (11, 676), (11, 745)]


def C(v=124): return Image.new("L", (S, S), v)


def rect(w, h, fill=40, bg=124):
    im = C(bg); ImageDraw.Draw(im).rectangle([(S - w) // 2, (S - h) // 2, (S + w) // 2, (S + h) // 2], fill=fill); return im


def step(frac=1.0, ang=0):
    a = np.full((S, S), 60, np.uint8); a[:S // 2] = 190
    if frac < 1: w = int(S * frac); l = (S - w) // 2; a[:, :l] = 124; a[:, l + w:] = 124
    return Image.fromarray(a).rotate(ang, fillcolor=124)


P = {"step full": step(), "step full a90": step(ang=90), "step 0.25": step(.25), "hbar 56": rect(S + 10, 56), "vbar 56": rect(56, S + 10),
     "dark sq 56": rect(56, 56), "dark sq 160": rect(160, 160), "rect 40x160": rect(40, 160), "rect 160x40": rect(160, 40),
     "light sq 96": rect(96, 96, 220), "sq 96 blur2": rect(96, 96).filter(ImageFilter.GaussianBlur(2)), "gray": C()}
names = list(P); imgs = [np.array(P[k].convert("RGB")) for k in names]
G._init(); rng = np.random.default_rng(1); pick = G._idx[rng.choice(len(G._idx), 200, replace=False)]
nat = np.asarray(G._X[np.sort(pick)])
zo = []
for im in nat:
    Q = Image.new("RGB", (S, S), tuple(int(v) for v in im.reshape(-1, 3).mean(0))); Q.paste(Image.fromarray(im).resize((112, 112)), (56, 56)); zo.append(np.array(Q))
Z = zmaps(np.stack(imgs + list(nat) + zo), T); n = len(names)
print("probe".ljust(16) + "".join(f"{b}n{k}".rjust(9) for b, k in T))
for i, k in enumerate(names): print(k.ljust(16) + "".join(f"{Z[t][i].max():9.2f}" for t in T))
print("200 random natural imgs, median max:".ljust(16) + "".join(f"{np.median(Z[t][n:n+200].max((1,2))):9.2f}" for t in T))
print("same, zoomed out (pasted frame):    " + "".join(f"{np.median(Z[t][n+200:].max((1,2))):9.2f}" for t in T))
