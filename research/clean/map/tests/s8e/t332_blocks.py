"""2 n332: is it a faint 8x8 block-grid (JPEG-style) detector on smooth ground? Prediction (after t332_local):
faint random 8x8 blocks (+-3 gray levels) on gray 158 drive it > 50% of natural max; 4x4 and 16x16 blocks much less;
a real JPEG of a faint smooth ramp > the same ramp without JPEG."""
import sys, os, io, json, numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from early_lib import *
rng = np.random.default_rng(0); MX, MN = 3.84, -4.97
def blocks(sz, amp, off=0, base=158):
    n = 224 // sz + 2; v = rng.uniform(-amp, amp, (n, n)).repeat(sz, 0).repeat(sz, 1)
    return np.repeat((base + v[off:off + 224, off:off + 224])[..., None], 3, -1)
def jpg(a, q):
    b = io.BytesIO(); Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(b, "JPEG", quality=q); return np.array(Image.open(b))
yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
ramp = np.repeat((150 + 16 * (xx + yy) / 448 + 3 * np.sin(xx / 23) * np.cos(yy / 31))[..., None], 3, -1)
S = {"flat158": patch((158,) * 3)}
for sz in (4, 8, 16):
    for amp in (1, 3, 8): S[f"blk{sz}-a{amp}"] = blocks(sz, amp)
S["blk8-a3-off4"] = blocks(8, 3, 4); S["blk8-a3-off2"] = blocks(8, 3, 2)
S["ramp"] = ramp
for q in (95, 75, 50, 30): S[f"ramp-jpg{q}"] = jpg(ramp, q)
S["ramp-jpg75-shift4"] = np.roll(jpg(ramp, 75), 4, 1)
for p, a in ((8, 3), (8, 1), (4, 3), (16, 3)):
    S[f"vgrat{p}-a{a}"] = 158 - a + 2 * a * (grating(90, p, .5, 0, 1).astype(np.float32))
    S[f"hgrat{p}-a{a}"] = 158 - a + 2 * a * (grating(0, p, .5, 0, 1).astype(np.float32))
# lone faint 8-px step lines (block edges only) vertical/horizontal
for o in ("v", "h"):
    a = np.full((224, 224), 158.); 
    for k in range(0, 224, 8):
        if o == "v": a[:, k] += 3
        else: a[k, :] += 3
    S[f"grid8-{o}"] = np.repeat(a[..., None], 3, -1)
X = np.stack([np.clip(v, 0, 255).astype(np.uint8) for v in S.values()]); Z = zmaps(2, 332, X)
k = 12; zz = Z[:, k:-k, k:-k]; out = {}
for j, n in enumerate(S):
    out[n] = (float(zz[j].max() / MX), float(zz[j].min() / abs(MN)), float(zz[j].std()))
    print(f"{n:20s} max {out[n][0]:+.2f}  min {out[n][1]:+.2f}  sd {out[n][2]:.2f}")
# periodicity in the map for blk8-a3: mean z by (row%2, col%2)
j = list(S).index("blk8-a3"); z = Z[j]
print("blk8-a3 mean by cell parity (r%2,c%2):", {(a, b): round(float(z[a::2, b::2].mean()), 2) for a in (0, 1) for b in (0, 1)})
j = list(S).index("ramp-jpg75"); z = Z[j]
print("ramp-jpg75 mean by cell parity:", {(a, b): round(float(z[a::2, b::2].mean()), 2) for a in (0, 1) for b in (0, 1)})
json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1)
