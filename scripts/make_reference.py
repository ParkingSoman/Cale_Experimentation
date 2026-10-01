"""Save bf16 reference predictions (top-64 log-probs per text position) for the held-out set."""
import os
import sys
import time

import mlx.core as mx
import numpy as np

from nest.drift import K, _text_positions
from nest.io import read_jsonl
from nest.stream import LayerwiseRunner

src = "models/gemma-4-12B-it-bf16"
out = sys.argv[1] if len(sys.argv) > 1 else "results/ref-bf16"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
os.makedirs(out, exist_ok=True)

recs = read_jsonl("data/heldout.jsonl")[:limit]
t = time.time()
runner = LayerwiseRunner(src)
samples = runner.prepare(recs)
print(f"prepared {len(samples)} samples, {sum(s['ids'].shape[1] for s in samples)} tokens, "
      f"{time.time() - t:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


def progress(i, s, h_in, h_out):
    if s is samples[-1]:
        print(f"layer {i} done {time.time() - t:.0f}s peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


runner.run(layer_hook=progress)
for s in samples:
    pos = _text_positions(s["ids"], s["kw"])
    lp = runner.logprobs(s, pos)
    top = mx.argpartition(-lp, K, axis=-1)[:, :K]
    vals = mx.take_along_axis(lp, top, axis=-1)
    mx.eval(top, vals)
    np.savez(f"{out}/{s['rec']['id']}.npz", pos=pos, idx=np.array(top).astype(np.int32),
             lp=np.array(vals).astype(np.float16))
print("done", time.time() - t)
