"""Build a "nest" model directory from the bf16 source, one tensor at a time.

Nest layout: the standard mlx-vlm layout, with MLP neurons reordered.
  gate_proj/up_proj rows and down_proj columns follow the importance order, so a
  narrower level uses a prefix of neurons. Quantization groups (64) run along the
  input axis, so any width that is a multiple of 64 is a valid quantized slice.
  (A transposed down_proj made slicing free but MLX's kernel for it is ~3x slower.)

Optional per-layer permutations reorder MLP neurons (most important first).
A permutation alone does not change the model's output.
"""
import json
import os
import shutil

import mlx.core as mx
import numpy as np

GROUP = 64
COPY_FILES = ["chat_template.jinja", "generation_config.json", "processor_config.json",
              "tokenizer.json", "tokenizer_config.json"]


def _quantizable(key, w):
    if not key.endswith(".weight") or w.ndim != 2:
        return False
    if "norm" in key or "pos_embedding" in key:
        return False
    return w.shape[-1] % GROUP == 0


def build(src, out, bits=8, perms=None, shard_gb=4.0):
    """perms: optional {layer_idx: np.ndarray of neuron indices, most important first}."""
    os.makedirs(out, exist_ok=True)
    idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    files = sorted(set(idx.values()))
    shard, shard_bytes, n_shard, weight_map = {}, 0, 0, {}

    def flush():
        nonlocal shard, shard_bytes, n_shard
        if not shard:
            return
        name = f"model-{n_shard:05d}.safetensors"
        mx.save_safetensors(f"{out}/{name}", shard, metadata={"format": "mlx"})
        for k in shard:
            weight_map[k] = name
        n_shard += 1
        shard, shard_bytes = {}, 0

    for fn in files:
        arrays = mx.load(f"{src}/{fn}")
        for key in sorted(arrays):
            w = arrays[key]
            if ".mlp." in key and ".layers." in key:
                layer = int(key.split(".layers.")[1].split(".")[0])
                p = None if perms is None or layer not in perms else mx.array(perms[layer])
                if p is not None:
                    w = w[:, p] if "down_proj" in key else w[p]   # reorder neurons
            if bits and _quantizable(key, w):
                q, s, b = mx.quantize(w, group_size=GROUP, bits=bits)
                base = key[: -len(".weight")]
                parts = {key: q, base + ".scales": s, base + ".biases": b}
            else:
                parts = {key: w}
            mx.eval(parts)
            for k, v in parts.items():
                shard[k] = v
                shard_bytes += v.nbytes
            if shard_bytes > shard_gb * 1e9:
                flush()
        del arrays
    flush()

    cfg = json.load(open(f"{src}/config.json"))
    if bits:
        cfg["quantization"] = {"group_size": GROUP, "bits": bits, "mode": "affine"}
        cfg["quantization_config"] = cfg["quantization"]
    cfg["nest_layout"] = {"down_proj": "standard", "permuted": perms is not None}
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=2)
    json.dump({"metadata": {}, "weight_map": weight_map},
              open(f"{out}/model.safetensors.index.json", "w"), indent=2)
    for f in COPY_FILES:
        if os.path.exists(f"{src}/{f}"):
            shutil.copy(f"{src}/{f}", f"{out}/{f}")
    return out


if __name__ == "__main__":
    import sys
    src, out, bits = sys.argv[1], sys.argv[2], int(sys.argv[3])
    perms = None
    if len(sys.argv) > 4:
        d = np.load(sys.argv[4])
        perms = {int(k.split("_")[1]): d[k] for k in d.files}
    build(src, out, bits, perms)
    print("built", out)
