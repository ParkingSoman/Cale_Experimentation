"""L3/L5 which-nodes-stay law: mode="full", cut=("mlp",), morder="root". See rules/l1_stream.py. Photo use: none."""
from rules import l1_stream as L

vit = L.make(mode="full", cut=("mlp",), morder="root")
