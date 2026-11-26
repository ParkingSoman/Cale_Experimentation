"""P5 r2: CLIP WHOLE cut with a per-part exponent of the weight x activity mix (see r2_p5_lib).
Function name = s<a stream>h<a heads>m<a mlp> in percent. s100h100m100 = the floor form on X (null endpoint).
Photo use: NONE."""
from rules import r2_p5_lib as L


def _mk(s, h, m):
    def fn(T, f, X=None):
        return L.build(T, f, X, 1.0, ("stream", "heads", "mlp"), {"stream": s / 100, "heads": h / 100, "mlp": m / 100})
    return fn


for _s in (0, 50, 75, 100):
    for _h in (0, 25, 50, 75, 100):
        for _m in (0, 25, 50, 75, 100):
            globals()[f"s{_s}h{_h}m{_m}"] = _mk(_s, _h, _m)
