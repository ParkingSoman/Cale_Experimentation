"""L4 identity build: kept read tilts toward carried deleted reads, tilt=0.3, on the L3 best (heads + MLP cut, head fold, roots, MLP drift). See rules/l1_stream.py. Photo use: none."""
from rules import l1_stream as L

vit = L.make(mode="full", hfold=True, morder="root", drift=("mlp",), tilt=0.3)
