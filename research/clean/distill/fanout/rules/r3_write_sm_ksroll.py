"""WRITE r3 variant sm_ksroll. See rules/r3_write_lib.py. Photo use: none."""
from rules import r3_write_lib as L

vit = L.make(cut=("stream", "mlp"), fold=("mlp",), kslice=True, null="roll", tag="sm_ksroll")
