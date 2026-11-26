"""P5 null split row: CLIP MLP neurons cut by the read-permuted null order, every other set at full width."""
from rules import p5rank as P

vit, _ = P.make(which="mlp", null=True)
