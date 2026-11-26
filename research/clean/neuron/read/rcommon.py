"""N10 step 2 shared parts: B, T, Base; bird photo sets; patch helpers.
Photo sets for birds (59 classes, the N5 list):
  choose = held birds (118) + spare birds (910: odd-position photos outside photos_test.npy; never used by
           train, held, cal or test, and never seen by T)
  score  = test birds (93), scored once per step.
Items: a survivor neuron j of block g = (W1[j], b1[j], W2[:, j]); a deleted neuron = Base neuron not in S (B order only:
it exists in neither B nor T as a neuron; "put back" = block g rebuilt with S + {j}, writes re-solved); a write bias b2 of block g."""
import copy, os, sys, json, time
import numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "gap"))
import common as C
from train_ref import load_mlp_state
OUT = os.path.join(HERE, "..", "results", "read"); CDIR = os.path.join(OUT, "cache"); os.makedirs(CDIR, exist_ok=True)
BIRDS = sorted(set(range(7, 25)) | set(range(80, 101)) | set(range(127, 147)))
LAB = C.net.Photos("read").lab

def spare_ix():
    sets = C.photo_sets(LAB)
    used = set(np.concatenate([sets[k] for k in ("train", "held", "cal", "test")]).tolist())
    return np.array([i for i in range(len(LAB)) if i not in used and LAB[i] in BIRDS])

def make_spare_cache():
    f = os.path.join(CDIR, "spare_x3.npy")
    if os.path.exists(f): return
    ix = spare_ix(); ph = C.net.Photos("read"); m = C.load("base")
    head = nn.Sequential(*[m.features[i] for i in range(5)])
    X = np.lib.format.open_memmap(f + ".tmp.npy", "w+", np.float16, (len(ix), 512, 14, 14)); Z = np.zeros((len(ix), 1000), np.float32)
    with torch.no_grad():
        for s in range(0, len(ix), 50):
            x3 = head(ph.tensor(ix[s:s + 50]).to(C.DEV)); X[s:s + 50] = x3.half().cpu().numpy(); Z[s:s + 50] = C.tail(m, x3).cpu().numpy()
    X.flush(); del X; os.rename(f + ".tmp.npy", f)
    np.save(os.path.join(CDIR, "spare_logits.npy"), Z); np.save(os.path.join(CDIR, "spare_ix.npy"), ix)

class RCache(C.Cache):
    def __init__(self, name):
        if name == "spare":
            self.x = np.load(os.path.join(CDIR, "spare_x3.npy"), mmap_mode="r"); self.z = np.load(os.path.join(CDIR, "spare_logits.npy"))
            self.ix = np.load(os.path.join(CDIR, "spare_ix.npy"))
        else: super().__init__(name)

class Rows:
    """A list of (cache, rows) for one photo set; eval returns per-photo KL to Base (concatenated)."""
    def __init__(self, parts): self.parts = parts
    def __len__(self): return sum(len(r) for _, r in self.parts)
    def kl(self, m):
        return np.concatenate([C.evaluate(m, c, rows_all=r)[1] for c, r in self.parts])

def bird_rows(c): return np.where(np.isin(LAB[c.ix], BIRDS))[0]

def photo_sets():
    make_spare_cache()
    held, spare, test = RCache("held"), RCache("spare"), RCache("test")
    return dict(choose=Rows([(held, bird_rows(held)), (spare, np.arange(len(spare)))]),
                held=Rows([(held, bird_rows(held))]), score=Rows([(test, bird_rows(test))]))

def nets():
    base = C.load("base"); cal = C.Cache("cal")
    calib = C.calibrate(base, (x for x, _ in cal.batches(25)))
    TB = torch.load(os.path.join(C.OUT, "T_build.pt"), weights_only=False); TT = torch.load(os.path.join(C.OUT, "T.pt"), weights_only=False)
    mB = copy.deepcopy(base); Ss = C.build(mB, calib, .5, "value"); load_mlp_state(mB, TB["state"])
    mT = copy.deepcopy(mB); load_mlp_state(mT, TT["state"])
    return base, mB, mT, [S.cpu() for S in Ss], calib

def L1(m, g): return C.blocks(m)[g].block[3]
def L2(m, g): return C.blocks(m)[g].block[5]

def set_neurons(dst, src, items):
    """Copy survivors' read row, read bias and write column from src into dst. items: list of (g, j) with j a position in the narrow layer."""
    with torch.no_grad():
        for g, j in items:
            for a, b in ((L1(dst, g), L1(src, g)),):
                a.weight[j].copy_(b.weight[j]); a.bias[j].copy_(b.bias[j])
            L2(dst, g).weight[:, j].copy_(L2(src, g).weight[:, j])

def set_wbias(dst, src, gs):
    with torch.no_grad():
        for g in gs: L2(dst, g).bias.copy_(L2(src, g).bias)

def set_block(dst, src, gs):
    with torch.no_grad():
        for g in gs:
            for a, b in ((L1(dst, g), L1(src, g)), (L2(dst, g), L2(src, g))):
                a.weight.copy_(b.weight); a.bias.copy_(b.bias)

def rebuild_block(m, base, g, S, calib, keep_from=None, keep_pos=None):
    """Block g of m := Base's block g shrunk to keep set S (Base indices), writes re-solved (the build's rule).
    keep_from/keep_pos: optionally copy survivors' trained numbers (positions keep_pos of keep_from's narrow layers) afterwards."""
    blk = C.blocks(m)[g]; bb = copy.deepcopy(C.blocks(base)[g]); mg, K = calib[g]
    C.shrink_block(bb, torch.sort(torch.as_tensor(S)).values, mg, K)
    blk.block[3], blk.block[5] = bb.block[3], bb.block[5]

def conv(o):
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    return o
def boot_ci(d, n=2000, seed=0):
    r = np.random.default_rng(seed); d = np.asarray(d)
    return np.percentile([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)], [2.5, 97.5]).tolist()
