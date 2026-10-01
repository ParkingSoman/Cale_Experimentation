"""Collect neuron and layer statistics from the bf16 12B on the calibration set.

Per layer, per MLP neuron j (a_j = gelu(gate_j(x)) * up_j(x)):
  sq_text[j], sq_img[j]   sum of a_j^2 over text / image tokens
  sum_text[j], sum_img[j] sum of a_j over text / image tokens
Per layer:
  cos     mean cosine similarity between layer input and output (ShortGPT "block influence" = 1 - cos)
  rel     mean ||out/scalar - in|| / ||in||, how much the layer changes the stream
Also the down_proj column norms, so neuron importance can be weighted by output size.

Output: results/stats-bf16.npz
"""
import time

import mlx.core as mx
import numpy as np
from mlx_vlm.models.gemma4 import language as g4

from nest.io import read_jsonl
from nest.stream import LayerwiseRunner

CUR = {"img": None}  # (1, L, 1) float mask of image tokens for the sample being processed
N_LAYERS, WIDTH = 48, 15360
acc = {k: np.zeros((N_LAYERS, WIDTH), np.float64) for k in ("sq_text", "sq_img", "sum_text", "sum_img")}
tok = {"text": 0, "img": 0}
layer_cos = np.zeros(N_LAYERS)
layer_rel = np.zeros(N_LAYERS)
layer_tokens = np.zeros(N_LAYERS)
down_norm = np.zeros((N_LAYERS, WIDTH), np.float32)
CUR_LAYER = {"i": 0}


def mlp_call(self, x):
    a = g4.geglu(self.gate_proj(x), self.up_proj(x))
    i = CUR_LAYER["i"]
    m = CUR["img"]
    af = a.astype(mx.float32)[0]
    if m is None:
        sq_t, s_t = (af * af).sum(0), af.sum(0)
        sq_i = s_i = None
    else:
        mi = m[0]
        mt = 1.0 - mi
        sq_t, s_t = (af * af * mt).sum(0), (af * mt).sum(0)
        sq_i, s_i = (af * af * mi).sum(0), (af * mi).sum(0)
    mx.eval(sq_t, s_t)
    acc["sq_text"][i] += np.array(sq_t); acc["sum_text"][i] += np.array(s_t)
    if sq_i is not None:
        mx.eval(sq_i, s_i)
        acc["sq_img"][i] += np.array(sq_i); acc["sum_img"][i] += np.array(s_i)
    return self.down_proj(a)


g4.MLP.__call__ = mlp_call

t = time.time()
recs = read_jsonl("data/calib.jsonl")
runner = LayerwiseRunner("models/gemma-4-12B-it-bf16")
samples = runner.prepare(recs)
for s in samples:
    tt = s["kw"].get("mm_token_type_ids")
    s["img"] = None if tt is None or int((tt != 0).sum()) == 0 else (tt != 0).astype(mx.float32)[..., None]
    n_img = 0 if s["img"] is None else int(s["img"].sum())
    tok["img"] += n_img
    tok["text"] += s["ids"].shape[1] - n_img
print(f"prepared {len(samples)} samples, tokens {tok}, {time.time() - t:.0f}s", flush=True)

orig_run_layer = runner.lm.layers[0].__class__.__call__


def hook(i, s, h_in, h_out):
    scalar = runner.lm.layers[i].layer_scalar
    a = h_in[0].astype(mx.float32)
    b = (h_out[0] / scalar).astype(mx.float32)
    cos = (a * b).sum(-1) / (mx.linalg.norm(a, axis=-1) * mx.linalg.norm(b, axis=-1) + 1e-6)
    rel = mx.linalg.norm(b - a, axis=-1) / (mx.linalg.norm(a, axis=-1) + 1e-6)
    mx.eval(cos, rel)
    layer_cos[i] += float(cos.sum()); layer_rel[i] += float(rel.sum()); layer_tokens[i] += cos.shape[0]
    if s is samples[-1]:
        print(f"layer {i} {time.time() - t:.0f}s", flush=True)


# Set the current sample / layer before each layer call.
LayerCls = runner.lm.layers[0].__class__
_call = LayerCls.__call__


def layer_call(self, x, mask=None, cache=None, per_layer_input=None, shared_kv=None, offset=None):
    CUR_LAYER["i"] = self.layer_idx
    return _call(self, x, mask, cache, per_layer_input, shared_kv, offset)


LayerCls.__call__ = layer_call
_orig_run = runner.run


def run_with_sample_masks():
    for i, layer in enumerate(runner.lm.layers):
        layer.update(runner._fresh(i))
        w = layer.mlp.down_proj.weight.astype(mx.float32)  # (hidden, inter)
        down_norm[i] = np.array(mx.linalg.norm(w, axis=0))
        for s in samples:
            CUR["img"] = s["img"]
            h_in = s["h"]
            h_out, _, _ = layer(h_in, s["masks"][layer.layer_type], None)
            mx.eval(h_out)
            hook(i, s, h_in, h_out)
            s["h"] = h_out
        layer.update(runner._fresh(i))
        mx.clear_cache()


run_with_sample_masks()
np.savez("results/stats-bf16.npz", **{k: v for k, v in acc.items()},
         n_text=tok["text"], n_img=tok["img"], cos=layer_cos / layer_tokens, rel=layer_rel / layer_tokens,
         down_norm=down_norm)
print("saved", time.time() - t)
