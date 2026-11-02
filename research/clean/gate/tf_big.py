"""TF-DIAG + TF-CTX on a well-trained LM (default SmolLM2-1.7B), parents from layer L-1 only, float32.
Same measures as tf_diag.py / tf_ctx.py and vision R2.  python3 tf_big.py L [K]"""
import glob, json, os, sys
import numpy as np
import mlx.core as mx
from mlx_lm import load
HERE = os.path.dirname(os.path.abspath(__file__)); CO = os.path.join(os.path.dirname(HERE), "coarse")
sys.path.insert(0, CO); sys.path.insert(0, os.path.join(CO, "methods")); sys.path.insert(0, os.path.dirname(HERE))
from contract import SEQ, ROOT  # noqa: E402
from tree_lib import token_blocks  # noqa: E402
import rcf_v1  # noqa: E402
from rcf_v1 import _capture  # noqa: E402


def _grab_call(self, x):
    self.x.append(np.array(x.astype(mx.float32)).reshape(-1, x.shape[-1])); return self.inner(x)


rcf_v1._Grab.__call__ = _grab_call

MODEL = os.environ.get("MODEL", "1.7B")
PATH = glob.glob(os.path.join(ROOT, "research/clean/hf/models--HuggingFaceTB--SmolLM2-1.7B/snapshots/*") if MODEL == "1.7B"
                 else os.path.join(ROOT, "research/tree/hf/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/*"))[0]
L = int(sys.argv[1]); NB = 24
model, tok = load(PATH); mx.eval(model.parameters())
cal = token_blocks(tok, "train", NB)
txt = open(os.path.join(ROOT, "research", "clean", "data", "code_train.txt")).read()
ids = np.array(tok.encode(txt[400000:1000000]), dtype=np.int32)[: NB * 513]
calm = mx.concatenate([cal, mx.array(ids.reshape(NB, 513))], 0)
lays = model.model.layers
cap = _capture(model, calm, {"x": (lays[L], "mlp"), "xp": (lays[L - 1], "mlp"), "s": (lays[L], "post_attention_layernorm")}, batch=2)
f32 = lambda a: np.array(a.astype(mx.float32))
silu = lambda z: z / (1 + np.exp(-z))
def dsilu(z): s = 1 / (1 + np.exp(-z)); return s * (1 + z * (1 - s))
mlp, mlpp = lays[L].mlp, lays[L - 1].mlp
G, U, D = f32(mlp.gate_proj.weight), f32(mlp.up_proj.weight), f32(mlp.down_proj.weight)
Gp, Up, Dp = f32(mlpp.gate_proj.weight), f32(mlpp.up_proj.weight), f32(mlpp.down_proj.weight)
N = G.shape[0]; K = int(sys.argv[2]) if len(sys.argv) > 2 else int(round(.03 * N))
X, Xp, S = cap["x"], cap["xp"], cap["s"]
Zg, Zu = X @ G.T, X @ U.T; H = silu(Zg) * Zu; Hp = silu(Xp @ Gp.T) * (Xp @ Up.T); del Xp
rms = np.sqrt((S.astype(np.float64) ** 2).mean(1) + 1e-5).astype(np.float32); gam = f32(lays[L].post_attention_layernorm.weight)
en = np.linalg.norm(H, axis=0) * np.linalg.norm(D, axis=0); rem = np.sort(np.argsort(-en)[K:])
MG, MU = (G * gam) @ Dp, (U * gam) @ Dp
ntop = max(1, int(.005 * len(H))); Pr = Hp / rms[:, None]
rng = np.random.default_rng(0); kids = rng.choice(rem, 200, replace=False)
modal, mass, attr = [], [], {}
for c in kids:
    t = np.argpartition(-np.abs(H[:, c]), ntop)[:ntop]
    A = MG[c] * ((dsilu(Zg[t, c]) * Zu[t, c])[:, None] * Pr[t]) + MU[c] * (silu(Zg[t, c])[:, None] * Pr[t])
    top = np.abs(A).argmax(1); modal.append(np.bincount(top).max() / len(t))
    mass.append(float(np.median(np.abs(A).max(1) / np.abs(A).sum(1)))); attr[int(c)] = np.abs(A.mean(0))
blk = np.repeat(np.arange(len(calm)), SEQ - 1); tr, te = blk % 2 == 0, blk % 2 == 1
Rk_te = H[te][:, kids] @ D[:, kids].T; Rk_tr = H[tr][:, kids] @ D[:, kids].T; den = (Rk_te.astype(np.float64) ** 2).sum()
Gtr = (Hp[tr].T @ Hp[tr]).astype(np.float64); Ctr = (H[tr][:, kids].T @ Hp[tr]).astype(np.float64)


def r2_top(n):
    A = np.zeros((len(kids), N))
    for i, c in enumerate(kids):
        ps = np.argsort(-attr[int(c)])[:n]; g = Gtr[np.ix_(ps, ps)] + np.eye(n) * 1e-3 * np.trace(Gtr) / N
        A[i, ps] = np.linalg.solve(g, Ctr[i, ps])
    Rh = (Hp[te] @ A.T.astype(np.float32)) @ D[:, kids].T; return float(1 - ((Rk_te - Rh).astype(np.float64) ** 2).sum() / den)


def r2_dense():
    best = -9; B_ = (Hp[tr].T @ Rk_tr).astype(np.float64)
    for lam in (1e-3, 1e-2, 1e-1, 1):
        W = np.linalg.solve(Gtr + np.eye(N) * lam * np.trace(Gtr) / N, B_).astype(np.float32)
        best = max(best, float(1 - ((Rk_te - Hp[te] @ W).astype(np.float64) ** 2).sum() / den))
    return best


out = {"model": "SmolLM2-" + MODEL, "L": L, "K": K, "n_neurons": N,
       "ctx": {"modal_parent_share_median": float(np.median(modal)), "top_parent_share_of_attr_mass_median": float(np.median(mass)), "uniform": 1 / N},
       "L-1": {f"top{n}": r2_top(n) for n in (1, 3, 10, 30, 100)}}
out["L-1"]["dense"] = r2_dense()
print(json.dumps(out))
p = os.path.join(ROOT, "results", "clean-tf-big.json"); res = json.load(open(p)) if os.path.exists(p) else {}
res[f"{MODEL}_L{L}"] = out; json.dump(res, open(p, "w"), indent=1)
