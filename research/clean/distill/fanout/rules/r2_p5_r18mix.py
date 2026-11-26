"""P5 r2: ResNet-18 whole cut, score = weight^(1-a) x activity^a per channel (inner and stream sets).
W = magcut.r18_scores (BN-folded weights = the inference map, no statistics as data), ACT = wide18.channel_scores on X.
a = 0 is the weight ranking, a = 1 the floor's activity ranking on X (endpoints of the same form).
Part changed: P5 Rank only. Plain cut (wide18.cut_from_a). Object (picked by): every ResNet inner and stream channel
(picked by the score). Photo use: NONE (weights + generic input X)."""
import torch

from rules import magcut, noisecut

_C = {}


def _scores(A, X):
    if "s" not in _C:
        import wide18
        A2 = A.to(noisecut.DEV).eval()
        with torch.no_grad():
            act = wide18.channel_scores(A2, X.to(noisecut.DEV).float(), bs=32)
        A.cpu()
        _C["s"] = (magcut.r18_scores(A), act)
    return _C["s"]


def _mix(w, x, a):
    return w.clamp_min(1e-12).log() * (1 - a) + x.clamp_min(1e-12).log() * a


def build(A, f, X, a_in, a_st):
    import wide18
    W, S = _scores(A, X)
    sc = dict(inner=[[_mix(W["inner"][s][b], S["inner"][s][b], a_in) for b in range(2)] for s in range(4)],
              stream=[_mix(W["stream"][s], S["stream"][s], a_st) for s in range(4)])
    with torch.no_grad():
        M, _ = wide18.cut_from_a(A.cpu(), f, sc)
    return M.cpu().eval()


def a0(A, f, X=None): return build(A, f, X, 0.0, 0.0)
def a25(A, f, X=None): return build(A, f, X, .25, .25)
def a50(A, f, X=None): return build(A, f, X, .5, .5)
def a75(A, f, X=None): return build(A, f, X, .75, .75)
def a100(A, f, X=None): return build(A, f, X, 1.0, 1.0)
