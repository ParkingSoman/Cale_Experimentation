"""Elastic Gemma 4: run any MLP width and any set of skipped layers on one loaded model.

Call patch() BEFORE loading a nest-layout model with nest.io.load().
Then use set_level(model, widths, skip) to switch levels with no reload.
"""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.gemma4 import language as g4

_ORIG_LAYER_CALL = g4.DecoderLayer.__call__


def _rows(lin, n):
    """First n output rows of a (possibly quantized) Linear, as matmul inputs."""
    if isinstance(lin, nn.QuantizedLinear):
        return lin.weight[:n], lin.scales[:n], lin.biases[:n], lin.group_size, lin.bits, lin.mode
    return lin.weight[:n], None, None, None, None, None


def _cols(lin, n, evaluate=True):
    """First n input columns of a (possibly quantized) Linear, copied into contiguous memory."""
    if isinstance(lin, nn.QuantizedLinear):
        per = 32 // lin.bits
        w = mx.contiguous(lin.weight[:, : n // per])
        s = mx.contiguous(lin.scales[:, : n // lin.group_size])
        b = mx.contiguous(lin.biases[:, : n // lin.group_size])
        if evaluate:
            mx.eval(w, s, b)
        return w, s, b, lin.group_size, lin.bits, lin.mode
    w = mx.contiguous(lin.weight[:, :n])
    if evaluate:
        mx.eval(w)
    return w, None, None, None, None, None


def _full(lin):
    if isinstance(lin, nn.QuantizedLinear):
        return lin.weight, lin.scales, lin.biases, lin.group_size, lin.bits, lin.mode
    return lin.weight, None, None, None, None, None


def _mm(x, parts):
    w, s, b, gs, bits, mode = parts
    if s is None:
        return x @ w.T
    return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits, mode=mode)


class ElasticMLP(nn.Module):
    """GeGLU MLP whose width can change at runtime (standard weight layout).

    gate/up use a prefix of rows (a free view). down_proj uses a prefix of
    columns, copied once per width into contiguous memory when the level is set.
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        H, I = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)
        self.full_width = I
        self.layer_idx = layer_idx
        # Plain-object state, so MLX does not treat it as model parameters.
        self.st = SimpleNamespace(width=I, comp=None, stats=None, down=None)

    # LEAN = True: slice down_proj on the fly in each call (low memory, a bit slower).
    # LEAN = False: keep one contiguous copy per level (fast; costs memory). Use for speed tests.
    LEAN = True

    def set_width(self, n):
        self.st.width = n
        self.st.down = None if n >= self.full_width or ElasticMLP.LEAN else _cols(self.down_proj, n)

    def __call__(self, x):
        st = self.st
        n = st.width
        if n >= self.full_width:
            a = g4.geglu(_mm(x, _full(self.gate_proj)), _mm(x, _full(self.up_proj)))
            down = _full(self.down_proj)
        else:
            a = g4.geglu(_mm(x, _rows(self.gate_proj, n)), _mm(x, _rows(self.up_proj, n)))
            down = st.down if st.down is not None else _cols(self.down_proj, n, evaluate=False)
        if st.stats is not None:
            st.stats(self.layer_idx, a)
        y = _mm(a, down)
        if st.comp is not None:
            y = y + st.comp
        return y


def _layer_call(self, x, mask=None, cache=None, per_layer_input=None, shared_kv=None, offset=None):
    if getattr(self, "_skip", False):
        # Skipped layer: keep the residual stream, but keep the layer's residual scale.
        return x * self.layer_scalar, None, offset
    return _ORIG_LAYER_CALL(self, x, mask, cache, per_layer_input, shared_kv, offset)


def patch():
    g4.MLP = ElasticMLP
    g4.DecoderLayer.__call__ = _layer_call


def layers(model):
    return model.language_model.model.layers


def set_level(model, widths=None, skip=(), comps=None):
    """widths: list of per-layer MLP widths (None = full). skip: layer indices to skip."""
    skip = set(skip)
    for i, layer in enumerate(layers(model)):
        mlp = layer.mlp
        mlp.set_width(mlp.full_width if widths is None else int(widths[i]))
        mlp.st.comp = None if comps is None else comps.get(i)
        layer._skip = i in skip
    mx.clear_cache()


def set_stats(model, fn):
    for layer in layers(model):
        layer.mlp.st.stats = fn
