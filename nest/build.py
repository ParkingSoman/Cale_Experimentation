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


def _rel(a, b):
    return (mx.linalg.norm((a - b).astype(mx.float32)) / mx.linalg.norm(b.astype(mx.float32))).item()


def _quant(w, bits, group, encoder):
    """encoder 'mlx': mx.quantize. 'grid': exact-grid encoder for QAT weights, used only
    where it beats mx.quantize on a sample of rows (it does not for reordered down_proj)."""
    if encoder == "grid" and bits == 4:
        from .gridq import quantize_grid
        sample = w[:512]
        e_grid = _rel(mx.dequantize(*quantize_grid(sample, group), group_size=group, bits=4), sample)
        e_mlx = _rel(mx.dequantize(*mx.quantize(sample, group_size=group, bits=4), group_size=group, bits=4), sample)
        if e_grid < e_mlx:
            return quantize_grid(w, group), "grid"
    return mx.quantize(w, group_size=group, bits=bits), "mlx"
COPY_FILES = ["chat_template.jinja", "generation_config.json", "processor_config.json",
              "tokenizer.json", "tokenizer_config.json"]


def _quantizable(key, w):
    if not key.endswith(".weight") or w.ndim != 2:
        return False
    if "norm" in key or "pos_embedding" in key:
        return False
    return w.shape[-1] % GROUP == 0


def build(src, out, bits=8, perms=None, shard_gb=4.0, overrides=None, encoder="mlx", group=GROUP):
    """perms: optional {layer_idx: np.ndarray of neuron indices, most important first}.
    overrides: optional {module_path_prefix: bits}, e.g. {"language_model.model.embed_tokens": 8}."""
    overrides = overrides or {}

    choices = {}

    def bits_for(key):
        for pre, b in overrides.items():
            if key.startswith(pre + "."):
                return b
        return bits

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
            kb = bits_for(key)
            if kb and _quantizable(key, w) and w.shape[-1] % group == 0:
                (q, s, b), how = _quant(w, kb, group, encoder)
                choices[how] = choices.get(how, 0) + 1
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
        cfg["quantization"] = {"group_size": group, "bits": bits, "mode": "affine"}
        for pre, b in overrides.items():
            cfg["quantization"][pre] = {"group_size": group, "bits": b, "mode": "affine"}
        cfg["quantization_config"] = cfg["quantization"]
    cfg["nest_layout"] = {"down_proj": "standard", "permuted": perms is not None, "encoder": encoder, "encoder_choices": choices}
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
    overrides = json.loads(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "-" else None
    encoder = sys.argv[6] if len(sys.argv) > 6 else "mlx"
    group = int(sys.argv[7]) if len(sys.argv) > 7 else GROUP
    if len(sys.argv) > 4 and sys.argv[4] != "-":
        d = np.load(sys.argv[4])
        perms = {int(k.split("_")[1]): d[k] for k in d.files}
    build(src, out, bits, perms, overrides=overrides, encoder=encoder, group=group)
    print("built", out)
