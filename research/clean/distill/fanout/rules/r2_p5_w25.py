"""P5 r2 PROMOTED rule (thin module): CLIP WHOLE cut. Stream and heads: the floor's activity ranking on X (a = 1).
MLP neurons: weight^.75 x activity^.25 (r2_p5_lib). Derived from: round-1 F1 + review (weights beat activity on MLP;
pink fires MLP at .35 of the photo rate, so activity alone misranks) -> a geometric mix. a = .25 was PICKED on 200
quick test photos (a fit); the held split row checks it. Part changed: P5 Rank only. Plain cut, no weight changed.
Object (picked by): every CLIP stream dim / head / MLP neuron (picked by the score). Photo use: NONE in the calc."""
from rules import r2_p5_lib as L


def vit(T, f, X=None):
    return L.build(T, f, X, 1.0, ("stream", "heads", "mlp"), {"mlp": .25})
