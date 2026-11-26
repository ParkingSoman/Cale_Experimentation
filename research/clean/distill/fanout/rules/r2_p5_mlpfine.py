"""P5 r2: CLIP MLP-only cut, finer exponent grid of weight^(1-a) x activity^a (see r2_p5_lib). Name = a in percent.
Also heads-only h75 (heads with some weight). a is chosen on quick TEST photos: a fit, checked on the HELD split."""
from rules import r2_p5_lib as L


def _m(a):
    return lambda T, f, X=None: L.build(T, f, X, a, ("mlp",))


m10, m15, m20, m30, m35 = _m(.10), _m(.15), _m(.20), _m(.30), _m(.35)


def h75(T, f, X=None):
    return L.build(T, f, X, .75, ("heads",))


def h90(T, f, X=None):
    return L.build(T, f, X, .90, ("heads",))
