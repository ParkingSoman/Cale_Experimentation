"""b9 n851: is the 'exact vertical = off' result an aliasing artefact? Bars drawn 4x supersampled (anti-aliased)."""
import sys, os, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_lib import maps
rng = np.random.default_rng(0); NOISE = np.clip(150 + rng.normal(0, 30, (896, 896, 1)).repeat(3, -1), 0, 255).astype(np.uint8)
def bar(a, aa=True, w=30, L=90):
    k = 4 if aa else 1; im = Image.fromarray(NOISE[:224 * k, :224 * k].copy()); d = ImageDraw.Draw(im)
    r = np.deg2rad(a); c, s = np.cos(r), np.sin(r); pts = [(-w / 2, -L), (w / 2, -L), (w / 2, L), (-w / 2, L)]
    d.polygon([(k * (112 + x * c - y * s), k * (112 + x * s + y * c)) for x, y in pts], fill=(40,) * 3)
    return np.array(im.resize((224, 224), Image.LANCZOS)) if aa else np.array(im)
S = {f"{'aa' if aa else 'px'}_tilt{a}": bar(a, aa) for aa in (False, True) for a in (0, 2, 5, 10, 20, 30, 60, 90)}
k = list(S); Z = maps(np.stack([S[x] for x in k]), [(9, 851)])[(9, 851)]
for i, x in enumerate(k): print(x.ljust(14), f"{Z[i][2:-2, 2:-2].max() / 3.82:+.2f}")
