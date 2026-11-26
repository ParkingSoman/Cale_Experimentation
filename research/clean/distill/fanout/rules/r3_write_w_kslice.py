"""WRITE r3 variant w_kslice. See rules/r3_write_lib.py. Photo use: none."""
from rules import r3_write_lib as L

vit = L.make(fold=("mlp",), kslice=True, tag="w_kslice")
