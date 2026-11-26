"""P5 r2: WHOLE cut where each part's units come from its own per-set greedy order (that part masked alone, others
full; see r2_p5_perset). Tests whether orders measured one part at a time combine better than the joint order.
Part changed: P5 Rank only. Plain cut. Object (picked by): every unit of every part (per-set greedy score).
Photo use: NONE."""
import torch

import vitgen
from rules import p5rank as P
from rules import r2_p5_perset as Q


def vit(T, f, X=None):
    C = {p: Q.order_part("clip", T, X, p) for p in ("stream", "heads", "mlp")}
    nb = len(T["blocks"])
    d, h, m = vitgen.sizes(f, T)
    return vitgen.cut(T, P.keep(C["stream"], "stream", d), [P.keep(C["heads"], f"head{i}", h) for i in range(nb)],
                      [P.keep(C["mlp"], f"mlp{i}", m) for i in range(nb)])


@torch.no_grad()
def r18(A, f, X=None):
    import wide18
    C = {p: Q.order_part("r18", A, X, p) for p in ("stream", "inner")}
    w = wide18.widths(f)
    sc = dict(stream=[], inner=[])
    for s in range(4):
        v = torch.zeros(wide18.BASE[s])
        v[P.keep(C["stream"], f"stream{s}", w[s])] = 1.0
        sc["stream"].append(v)
        row = []
        for b in range(2):
            v = torch.zeros(wide18.BASE[s])
            v[P.keep(C["inner"], f"inner{s}{b}", w[s])] = 1.0
            row.append(v)
        sc["inner"].append(row)
    S, _ = wide18.cut_from_a(A.cpu(), f, sc)
    return S.cpu().eval()
