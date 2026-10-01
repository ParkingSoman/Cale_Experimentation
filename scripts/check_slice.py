"""Check one layer: elastic 8-bit slice vs. bf16 MLP restricted to the same neurons."""
import json, sys
import mlx.core as mx, mlx.nn as nn, numpy as np
sys.path.insert(0, ".")
from nest.elastic import _rows, _cols, _mm

def get(d, keys):
    idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
    return {k: mx.load(f"{d}/{idx[k]}")[k] for k in keys}

perm = np.load("results/perm-wanda.npz")
x = mx.random.normal((8, 3840)).astype(mx.bfloat16)
for i in (5, 20, 40):
    p = f"language_model.model.layers.{i}.mlp."
    bf = get("models/gemma-4-12B-it-bf16", [p + n + "_proj.weight" for n in ("gate", "up", "down")])
    q = get("models/nest8", [p + n + "_proj." + s for n in ("gate", "up", "down") for s in ("weight", "scales", "biases")])
    def ql(n):
        l = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=8)
        l.weight, l.scales, l.biases = q[p+n+"_proj.weight"], q[p+n+"_proj.scales"], q[p+n+"_proj.biases"]
        return l
    G, U, D = ql("gate"), ql("up"), ql("down")
    for n in (15360, 13824, 7680):
        keep = mx.array(perm[f"layer_{i}"][:n])
        a = nn.gelu_approx(x @ bf[p+"gate_proj.weight"][keep].T) * (x @ bf[p+"up_proj.weight"][keep].T)
        ref = a @ bf[p+"down_proj.weight"][:, keep].T
        full_a = nn.gelu_approx(x @ bf[p+"gate_proj.weight"].T) * (x @ bf[p+"up_proj.weight"].T)
        full = full_a @ bf[p+"down_proj.weight"].T
        e = nn.gelu_approx(_mm(x, _rows(G, n))) * _mm(x, _rows(U, n))
        out = _mm(e, _cols(D, n))
        r = lambda u, v: (mx.linalg.norm((u - v).astype(mx.float32)) / mx.linalg.norm(v.astype(mx.float32))).item()
        print(f"layer {i} width {n}: elastic vs bf16-same-neurons {r(out, ref):.4f} | bf16-subset vs bf16-full {r(ref, full):.4f}")
