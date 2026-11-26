"""WRITE r3 variant w_self. See r3_write_lib.py. Photo use: none."""
from rules import r3_write_lib as L

vit = L.make(fold=("mlp",), xfold="distill/fanout/cache/clip_self_n256_k32_s1.pt", tag="w_self")
