"""Top-1 accuracy (true labels) and agreement with the intact model on the sealed 3k: intact, drop block 16, and trained
block-17 snapshots (d5cache npz files given as args)."""
import os, sys, copy, numpy as np, mlx.core as mx
from convnext import *
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); z = np.load(os.path.join(D, "x16.npz")); yE = z["yE"]; LE = z["LE"]
def run(blk):
    out = []
    for a in range(0, 3000, 200): out.append(np.array(m.head(blk(mx.array(z["XE"][a:a + 200]).astype(mx.float32)))))
    return np.concatenate(out)
ti = LE.argmax(1); print(f"intact: top-1 {(ti == yE).mean():.4f}")
def rep(nm, L): p = L.argmax(1); print(f"{nm:60s} top-1 {(p == yE).mean():.4f}  agree w/ intact {(p == ti).mean():.4f}")
rep("drop block 16", run(m.blocks[17]))
for f in sys.argv[1:]:
    b = copy.deepcopy(m.blocks[17]); w = np.load(os.path.join(D, f)); b.load_weights([(k.replace("blocks.17.", ""), mx.array(v)) for k, v in w.items() if k.startswith("blocks.17.") or not k.startswith("blocks.")], strict=False); rep(f, run(b))
