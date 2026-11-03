# serves: H-map
"""The microscope's exact sums. Every kind of node must sum to its true value, and a fill computed by the sum must match a real forward pass."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ev2 import below, exact_inputs, grid, mean_of, name, parse, top_photos  # noqa: E402
from net import Photos, block_parts, forward, load, stages, stem  # noqa: E402

torch.set_num_threads(2)
m = load()
P = Photos("read")
for text in ("s1b0:3", "s1b0i:3", "s1b1:7", "s1b1i:7", "s2b0:5", "s2b0i:5", "s2b1:9", "s3b0:11", "s3b1i:11", "s4b0:179", "s4b1:179", "s4b1i:179"):
    n = parse(text)
    idx, pos = top_photos(n, "top", 8)
    D, names, base, true = exact_inputs(m, P.tensor(idx), n, pos)
    miss = float(np.abs(base + D.sum(1) - true).max() / (np.abs(true).mean() + 1e-6))
    assert miss < 2e-3, (text, miss)
    assert len(names) == D.shape[1]
    print(f"{name(n)}: {D.shape[1]} inputs, sum misses by {miss:.1e} of the value")

# one input filled by hand, through the real layer, equals the sum's prediction
n = parse("s4b1:179")
idx, pos = top_photos(n, "top", 8)
x = P.tensor(idx)
D, names, base, true = exact_inputs(m, x, n, pos)
j = int(np.argmax(np.abs(D[:, :-1]).mean(0)))
with torch.no_grad():
    _, o = forward(m, x)
    bl = stages(m)[4][1]
    inner = o[(4, 1)]["inner"].clone()
    inner[:, j] = mean_of("s4b1_im")[j]
    z = (bl.bn2(bl.conv2(inner)) + o[(4, 0)]["h"])[:, 179].flatten(1)[torch.arange(8), torch.as_tensor(pos)].numpy()
miss = float(np.abs((true - z) - D[:, j]).max())
assert miss < 1e-3, miss
print(f"one inner channel filled by hand, against the sum: misses by {miss:.1e}")
print("ok")
