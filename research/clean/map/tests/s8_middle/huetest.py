"""b8 n1026 colour prediction: on the target's 30 most negative images, does swapping warm->cool (R<->B) or gray
weaken the negative pole? Same edit on the 30 top images for the positive pole. Also for candidate units, measured
at the target's spots. Usage: python huetest.py TB,TN [B,N ...]"""
import sys, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "vision2")); sys.path.insert(0, os.path.join(H, "..", "s6"))
import gallery as G
from judge_lib import maps
U = [tuple(map(int, a.split(","))) for a in sys.argv[1:]]; T = U[0]
G._init(); S = G.scan({T[0]: [T[1]]}); vm, pm, vn, pn = S[T]
ED = {"orig": lambda x: x, "swapRB": lambda x: x[..., ::-1], "gray": lambda x: np.repeat(x.mean(-1, keepdims=True), 3, -1).astype(np.uint8)}
def pool(z):
    f = z.shape[1] // 14; return z if f == 1 else z.reshape(len(z), 14, f, 14, f).mean((2, 4))
for pole, o in (("pos", np.argsort(-vm)[:30]), ("neg", np.argsort(vn)[:30])):
    imgs = np.stack([np.asarray(G._X[G._idx[i]]) for i in o])
    Z = {e: maps(f(imgs), U) for e, f in ED.items()}
    cells = [(np.argmax if pole == "pos" else np.argmin)(z) for z in Z["orig"][T]]
    for u in U:
        vals = {e: np.mean([pool(Z[e][u])[j].ravel()[cells[j]] for j in range(30)]) for e in ED}
        print(f"{pole} b{u[0]}n{u[1]}: " + "  ".join(f"{e} {v:+.2f}" for e, v in vals.items()))
