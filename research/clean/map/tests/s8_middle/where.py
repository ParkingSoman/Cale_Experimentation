"""Is the neuron a frame-position ("where") unit? Mean 14x14 map over 300 natural images, on uniform gray,
pink noise, and on rolled images (content shifted 64 px right: a frame unit keeps its peak column, a content unit
moves it). Usage: python where.py B,N [B,N ...]"""
import sys, os, json
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2")); sys.path.insert(0, os.path.join(H, "..", "s6"))
import gallery as G
from judge_lib import maps
T = [tuple(map(int, a.split(","))) for a in sys.argv[1:]]
G._init(); rng = np.random.default_rng(0); sel = rng.choice(len(G._idx), 300, replace=False)
nat = np.stack([np.asarray(G._X[G._idx[i]]) for i in sel])
gray = np.stack([np.full((224, 224, 3), v, np.uint8) for v in (40, 128, 210)])
def pink(seed):
    r = np.random.default_rng(seed); f = np.fft.fftfreq(224)[:, None] ** 2 + np.fft.fftfreq(224)[None] ** 2; f[0, 0] = 1
    out = []
    for c in range(3):
        x = np.real(np.fft.ifft2(np.fft.fft2(r.standard_normal((224, 224))) / f ** .5)); out.append((x - x.mean()) / x.std())
    return np.clip(128 + 40 * np.stack(out, -1), 0, 255).astype(np.uint8)
noise = np.stack([pink(s) for s in range(20)])
roll = np.roll(nat[:100], 64, axis=2)
M = {}
for name, arr in (("nat", nat), ("gray", gray), ("pink", noise), ("roll", roll), ("natsub", nat[:100])):
    Z = maps(arr, T)
    for t in T: M.setdefault(t, {})[name] = Z[t]
np.set_printoptions(linewidth=200, precision=1, suppress=True)
for t in T:
    m = M[t]; print(f"\n===== b{t[0]} n{t[1]}")
    for name in ("nat", "gray", "pink"):
        a = m[name].mean(0); print(f"-- mean map {name} (min {a.min():.2f} max {a.max():.2f})"); print(a)
    # where does the per-image peak go when content is rolled 64 px (4.6 cells) right?
    pk = lambda Z: np.array([divmod(int(np.argmax(z)), z.shape[1]) for z in Z])
    p0, p1 = pk(m["natsub"]), pk(m["roll"])
    dc = p1[:, 1] - p0[:, 1]; same = (np.abs(dc) <= 1).mean(); moved = (np.abs(dc - 4.6) <= 1.2).mean()
    print(f"roll test: peak col same {same:.2f}, moved with content {moved:.2f}; peak value ratio roll/orig {np.median(m['roll'].max((1,2))/np.maximum(m['natsub'].max((1,2)),1e-3)):.2f}")
    print(f"gray max {m['gray'].max():.2f} at", [divmod(int(np.argmax(z)), 14) for z in m['gray']], f"; pink max mean {m['pink'].max((1,2)).mean():.2f}; nat max mean {m['nat'].max((1,2)).mean():.2f}")
