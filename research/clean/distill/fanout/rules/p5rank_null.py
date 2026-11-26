"""P5 null (read-permuted): the p5rank formula with the read pairing permuted inside each set (see p5rank.py).
CLIP order stops at f = .5. Photo use: none."""
from rules import p5rank as P

vit, r18 = P.make(null=True)
