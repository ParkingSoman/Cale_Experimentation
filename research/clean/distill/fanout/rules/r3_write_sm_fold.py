"""WRITE r3 variant sm_fold. See r3_write_lib.py. Photo use: none."""
from rules import r3_write_lib as L

vit = L.make(cut=("stream", "mlp"), fold=("mlp",), tag="sm_fold")
