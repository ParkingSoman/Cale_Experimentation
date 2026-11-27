"""L3 depth: cut only blocks (8, 9, 10, 11) (heads + MLP, head fold, roots, MLP drift). See rules/l1_stream.py. Photo use: none."""
from rules import l1_stream as L

vit = L.make(mode="full", hfold=True, morder="root", drift=("mlp",), blocks=(8, 9, 10, 11))
