"""Sanity checks before any result is trusted."""
import time
import numpy as np
import mlx.core as mx
from tree_lib import *

t0 = time.time()
model, tok = load_model()
cal = token_blocks(tok, "train", 32)
held = token_blocks(tok, "test", 100)
print("blocks", cal.shape, held.shape, f"{time.time()-t0:.1f}s")

base = eval_loss(model, held)
print("baseline held-out CE", base)

W = Weights(model)

# 1. Rebuild every layer unchanged through the merge code (each neuron its own group)
X = capture_mlp_inputs(model, cal)
for i in range(len(W.G)):
    s = canonical_sign(X[i], W.G[i], W.U[i])
    lab = np.arange(W.G.__len__() and W.G[i].shape[0])
    G2, U2, D2 = merge_groups(W.G[i], W.U[i], W.D[i], s, lab)
    W.set_layer(i, G2, U2, D2)
W.commit()
print("identity rebuild", eval_loss(model, held, teacher=None)["ce"], "(must equal baseline)")
W.restore()

# 2. Duplicate a neuron, split its output in half, merge the two back: must be exact
i = 5
G, U, D = W.G[i], W.U[i], W.D[i]
G2 = np.concatenate([G, G[:1]]); U2 = np.concatenate([U, U[:1]])
D2 = np.concatenate([D, D[:, :1]], 1); D2[:, 0] *= 0.5; D2[:, -1] *= 0.5
Xi = X[i][:2000]
y1 = hidden(Xi, G, U) @ D.T
y2 = hidden(Xi, G2, U2) @ D2.T
print("split is exact, max abs diff", float(np.abs(y1 - y2).max()))
lab = np.arange(G2.shape[0]); lab[-1] = 0
s = np.ones(G2.shape[0], np.float32)
G3, U3, D3 = merge_groups(G2, U2, D2, s, lab)
y3 = hidden(Xi, G3, U3) @ D3.T
print("merge back is exact, max abs diff", float(np.abs(y1 - y3).max()))

# 3. Sign flip: flipping (up row, down column) must not change the neuron
j = 7
flip = np.ones(G.shape[0], np.float32); flip[j] = -1
y4 = hidden(Xi, G, U * flip[:, None]) @ (D * flip[None, :]).T
print("sign flip is the same function, max abs diff", float(np.abs(y1 - y4).max()))

# 4. LS refit of the full layer onto itself must be ~identity
Y = hidden(X[i], G, U) @ D.T
D5 = ls_down(X[i], G, U, Y)
print("LS refit rel err on full layer", rel_err(Y, hidden(X[i], G, U) @ D5.T))
print(f"done {time.time()-t0:.1f}s")
