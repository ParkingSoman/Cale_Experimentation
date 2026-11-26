"""WRITE r3 variant w_ksroll. See rules/r3_write_lib.py. Photo use: none."""
from rules import r3_write_lib as L

vit = L.make(fold=("mlp",), kslice=True, null="roll", tag="w_ksroll")
