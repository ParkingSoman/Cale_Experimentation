"""P5 split row: CLIP MLP neurons cut by the p5rank order, every other set at full width."""
from rules import p5rank as P

vit, _ = P.make(which="mlp")
