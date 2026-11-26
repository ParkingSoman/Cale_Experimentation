"""P5 split row: stream cut by the p5rank order, every other set at full width (CLIP: cut; ResNet: reads zeroed as in _split)."""
from rules import p5rank as P

vit, r18 = P.make(which="stream")
