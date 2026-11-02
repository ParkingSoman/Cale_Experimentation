"""Synthetic tests for the PREDICTIONS.md table. Prints 14x14-map readings at named spots. Optional extra units
(B,N) get the same stimuli (for candidate checks): python synth.py [B,N ...]"""
import sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "s6"))
from judge_lib import maps
EX = [tuple(map(int, a.split(","))) for a in sys.argv[1:]]
def canvas(v=128): return Image.new("RGB", (224, 224), (v, v, v))
def arr(im): return np.asarray(im, np.uint8)
S = {}
# b6: small marks
for name, col in (("dark", (20, 20, 20)), ("red", (200, 20, 20))):
    im = canvas(170); d = ImageDraw.Draw(im); d.rectangle([106, 106, 118, 118], fill=col); S[f"b6_sq12_{name}"] = im
    im = canvas(170); d = ImageDraw.Draw(im); d.rectangle([109, 92, 115, 132], fill=col); S[f"b6_vbar_{name}"] = im
    im = canvas(170); d = ImageDraw.Draw(im); d.rectangle([92, 109, 132, 115], fill=col); S[f"b6_hbar_{name}"] = im
    S[f"b6_sq12_{name}_blur3"] = S[f"b6_sq12_{name}"].filter(ImageFilter.GaussianBlur(3))
    S[f"b6_vbar_{name}_blur3"] = S[f"b6_vbar_{name}"].filter(ImageFilter.GaussianBlur(3))
    im = canvas(170); d = ImageDraw.Draw(im); d.rectangle([64, 64, 160, 160], fill=col); S[f"b6_block96_{name}"] = im
# b7: triangles
tri = [(112, 50), (60, 170), (164, 170)]
im = canvas(); ImageDraw.Draw(im).polygon(tri, fill=(30, 30, 30)); S["b7_tri_up"] = im; S["b7_tri_down"] = im.rotate(180)
im = canvas(); ImageDraw.Draw(im).ellipse([52, 52, 172, 172], fill=(30, 30, 30)); S["b7_disc"] = im
im = canvas(); ImageDraw.Draw(im).polygon(tri, fill=(220, 220, 220)); S["b7_tri_up_light"] = im
# b9: dark rectangle on light
im = canvas(190); ImageDraw.Draw(im).rectangle([40, 70, 184, 150], fill=(30, 30, 30)); S["b9_darkrect"] = im
S["b9_darkrect_inv"] = Image.fromarray(255 - arr(im))
# b10: discs
im = canvas(); ImageDraw.Draw(im).ellipse([52, 52, 172, 172], fill=(25, 25, 25)); S["b10_disc120"] = im
im = canvas(); ImageDraw.Draw(im).ellipse([97, 97, 127, 127], fill=(25, 25, 25)); S["b10_disc30"] = im
im = canvas(); ImageDraw.Draw(im).rectangle([52, 52, 172, 172], fill=(25, 25, 25)); S["b10_square120"] = im
S["b10_disc120_soft"] = S["b10_disc120"].filter(ImageFilter.GaussianBlur(6))
im = canvas(25); S["b10_flatdark"] = im
names = list(S); X = np.stack([arr(S[k]) for k in names])
T = [(6, 1286), (7, 1155), (9, 64), (10, 95)] + EX
Z = maps(X, T); np.set_printoptions(linewidth=200, precision=1, suppress=True)
MX = {(6, 1286): 3.52, (7, 1155): 4.09, (9, 64): 3.03, (10, 95): 4.07}
def cell(y, x): return int(y // 16), int(x // 16)
for t in T:
    print(f"\n=== b{t[0]} n{t[1]}")
    for i, k in enumerate(names):
        z = Z[t][i]
        if z.shape[0] != 14:  # pool early-block maps to 14
            f = z.shape[0] // 14; z = z.reshape(14, f, 14, f).max((1, 3))
        r, c = divmod(int(np.argmax(z)), 14); rn, cn = divmod(int(np.argmin(z)), 14)
        print(f"{k:22s} max {z.max():+.2f} at ({r:2d},{c:2d})  min {z.min():+.2f} at ({rn:2d},{cn:2d})  centre {z[7,7]:+.2f}")
    if t in MX: print("max ref", MX[t])
# detail maps for key stimuli
for t, ks in {(7, 1155): ["b7_tri_up", "b7_tri_down"], (9, 64): ["b9_darkrect", "b9_darkrect_inv"], (10, 95): ["b10_disc120"]}.items():
    for k in ks: print(f"\n-- b{t[0]} n{t[1]} map {k}"); print(Z[t][names.index(k)])
