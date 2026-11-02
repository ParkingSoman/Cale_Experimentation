"""b2 n194 test P2: line ends. Half lines in 8 directions (end at centre), full lines, segments, corners, dots,
end of a bar of several widths, rows of parallel short strokes. Report max/neuron max and peak distance to centre."""
import sys, os, json, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_lib import maps
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
C = json.load(open(os.path.join(D, "s6", "candidates.json")))["2_194"]
tg = [(2, 194)] + [tuple(v) for v in C.values()]; MX = {t: json.load(open(os.path.join(D, "deep", f"b{t[0]}_n{t[1]}.json")))["max"] for t in tg}


def canvas(bg=128): im = Image.new("RGB", (224, 224), (bg,) * 3); return im, ImageDraw.Draw(im)


def seg(ang, L, w=3, fg=30, start=0):
    im, d = canvas(); a = np.deg2rad(ang); x0, y0 = 112 + np.cos(a) * start, 112 - np.sin(a) * start
    d.line([(x0, y0), (112 + np.cos(a) * L, 112 - np.sin(a) * L)], fill=(fg,) * 3, width=w); return np.array(im)


S = {}
for ang in range(0, 360, 45): S[f"half_dir{ang:03d}"] = seg(ang, 150)       # line from centre going toward ang (end at centre)
for ang in (0, 45, 90, 135): S[f"full_{ang}"] = seg(ang, 150, start=-150)
for L in (4, 8, 16, 32): S[f"segment_h_len{2*L}"] = seg(0, L, start=-L)
for w in (1, 2, 3, 5, 8, 12): S[f"half_dir180_w{w}"] = seg(180, 150, w=w)
S["half_dir180_light"] = seg(180, 150, fg=230)
im, d = canvas(); d.line([(112, 112), (0, 112)], fill=(30,) * 3, width=3); d.line([(112, 112), (112, 0)], fill=(30,) * 3, width=3); S["corner_L_left_up"] = np.array(im)
im, d = canvas(); d.line([(0, 112), (224, 112)], fill=(30,) * 3, width=3); d.line([(112, 112), (112, 0)], fill=(30,) * 3, width=3); S["T_junction"] = np.array(im)
im, d = canvas(); d.ellipse([108, 108, 116, 116], fill=(30,) * 3); S["dot8"] = np.array(im)
im, d = canvas(); d.rectangle([0, 0, 112, 224], fill=(40,) * 3); S["edge_vertical"] = np.array(im)
im, d = canvas(); d.rectangle([0, 90, 112, 134], fill=(40,) * 3); S["bar_end_h44"] = np.array(im)
im, d = canvas()
for x in range(40, 190, 6): d.line([(x, 90), (x, 134)], fill=(30,) * 3, width=2)
S["parallel_strokes_p6"] = np.array(im)
im, d = canvas()
for x in range(40, 190, 6): d.line([(x, 0), (x, 224)], fill=(30,) * 3, width=2)
S["parallel_full_p6"] = np.array(im)
keys = list(S); imgs = np.stack([S[k] for k in keys]); Z = maps(imgs, tg); res = {}
print("stimulus".ljust(22) + "  T194 peak(px from centre)  " + " ".join(f"{c:>6s}" for c in C))
for i, k in enumerate(keys):
    z = Z[(2, 194)][i, 8:48, 8:48]; r, c = np.unravel_index(z.argmax(), z.shape); dist = np.hypot((r + 8 + .5) * 4 - 112, (c + 8 + .5) * 4 - 112)
    row = [float(Z[t][i][int(Z[t].shape[1]*.15):int(Z[t].shape[1]*.85), int(Z[t].shape[1]*.15):int(Z[t].shape[1]*.85)].max() / MX[t]) for t in tg]
    res[k] = row + [float(dist)]; print(k.ljust(22) + f"{row[0]:+6.2f}  {dist:5.0f}               " + " ".join(f"{v:+6.2f}" for v in row[1:]))
json.dump({"cands": list(C), "res": res}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "t194_ends.json"), "w"), indent=1)
