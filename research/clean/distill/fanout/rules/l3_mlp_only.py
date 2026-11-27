"""L3 heads test: mode="full", cut=("mlp",). Stream at full width. See rules/l1_stream.py. Photo use: none."""
from rules import l1_stream as L

vit = L.make(mode="full", cut=("mlp",))
