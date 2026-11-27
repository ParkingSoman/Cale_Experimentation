"""Span read for L1 test 4: spectrum of G^1/2 C G^1/2 (all vs kept-only) at f = .95, .85, .5, .25. Weights + pink X."""
import sys, os, json, torch
FAN = "/Users/shonusengupta/Cale_Experimentation/research/clean/distill/fanout"
sys.path.insert(0, FAN); sys.path.insert(0, os.path.dirname(FAN))
from rules import l1_stream as L, r2_p6_core as P6, r3_joint_clip as J
from rules import p4write as P4
import board, vitgen
T = board.clip_teacher()
X = P4._x_vit(torch.load(os.path.join(FAN, "cache", "input_pink_n256_s1.pt"), weights_only=True).float())
D = T["cls"].numel()
m = P6.site_moments(T, X)
Mall = m["M"]
Call = (Mall / Mall.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None]).sum(0)
Gall = sum(G / G.trace() for G in P6.readers(T))
def spec(G, C):
    Gh = P6._sym_sqrt(G)
    e = torch.linalg.eigvalsh(Gh @ C @ Gh).flip(0).clamp_min(0)
    c = e.cumsum(0) / e.sum()
    return e, c
def dims(c, q):
    return int((c < q).sum()) + 1
out = {}
e, c = spec(Gall, Call)
out["all"] = dict(d90=dims(c, .9), d99=dims(c, .99), d999=dims(c, .999))
print("all readers/writers:", out["all"])
for f in (.95, .85, .5, .25):
    _, hk, mk = J.keep_units(T, f, X, "act", "w", 0)
    d, _, _ = vitgen.sizes(f, T)
    Gk = sum(G / G.trace() for G in L.kept_readers(T, hk, mk))
    Mk = L.kept_written(T, X, hk, mk)
    Ck = (Mk / Mk.diagonal(dim1=1, dim2=2).sum(-1)[:, None, None]).sum(0)
    ek, ck = spec(Gk, Ck)
    r = dict(f=f, d_cut=d, kept_d90=dims(ck, .9), kept_d99=dims(ck, .99), kept_d999=dims(ck, .999),
             kept_lost_at_dcut=float(1 - ck[d - 1]), all_lost_at_dcut=float(1 - c[d - 1]))
    out[str(f)] = r
    print(r, flush=True)
json.dump(out, open(os.path.join(FAN, "out", "l1_span.json"), "w"), indent=1)
