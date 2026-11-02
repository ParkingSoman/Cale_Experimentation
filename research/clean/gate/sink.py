"""Sink guard: keep the ORIGINAL layer output at block position 0 (the attention sink). In Cale, position 0 is always
the same chat-template token, so this output is a constant per layer."""
import mlx.core as mx
import mlx.nn as nn


class SinkGuard(nn.Module):
    def __init__(self, new, orig):
        super().__init__(); self.new, self.orig = new, orig

    def __call__(self, x):
        y = self.new(x)
        if x.ndim == 3:
            y0 = self.orig(x[:, :1])
            y = mx.concatenate([y0, y[:, 1:]], axis=1)
        return y

    def units(self, x):
        return self.new.units(x)

    def n_params(self):
        return self.new.n_params()
