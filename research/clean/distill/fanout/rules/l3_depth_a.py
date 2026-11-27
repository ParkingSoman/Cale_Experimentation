"""L3 depth: cut only blocks (0, 1, 2, 3) (heads + MLP, head fold, roots, MLP drift). See rules/l1_stream.py. Photo use: none."""
from rules import l1_stream as L

vit = L.make(mode="full", hfold=True, morder="root", drift=("mlp",), blocks=(0, 1, 2, 3))
