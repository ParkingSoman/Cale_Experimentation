"""WRITE r3 variant sm_self. See r3_write_lib.py. Photo use: none."""
from rules import r3_write_lib as L

vit = L.make(cut=("stream", "mlp"), fold=("mlp",), xfold="distill/fanout/cache/clip_self_n256_k32_s1.pt", tag="sm_self")
