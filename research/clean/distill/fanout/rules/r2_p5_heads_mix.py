"""P5 r2: CLIP heads-only and stream-only cuts with the weight x activity mix (see r2_p5_lib). a in the name."""
from rules import r2_p5_lib as L


def h0(T, f, X=None): return L.build(T, f, X, 0.0, ("heads",))
def h50(T, f, X=None): return L.build(T, f, X, 0.5, ("heads",))
def h100(T, f, X=None): return L.build(T, f, X, 1.0, ("heads",))
def s50(T, f, X=None): return L.build(T, f, X, 0.5, ("stream",))
def s100(T, f, X=None): return L.build(T, f, X, 1.0, ("stream",))
