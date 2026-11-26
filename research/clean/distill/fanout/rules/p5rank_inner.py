"""P5 split row: ResNet inner channels cut by the p5rank order (conv2 columns zeroed as in _split), the stream at full width."""
from rules import p5rank as P

_, r18 = P.make(which="inner")
