"""WRITE r3 variant hm_fold. See r3_write_lib.py. Photo use: none."""
from rules import r3_write_lib as L

vit = L.make(cut=("heads", "mlp"), fold=("mlp",), tag="hm_fold")
