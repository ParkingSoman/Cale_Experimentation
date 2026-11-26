"""WRITE r3 variant w_fold. See r3_write_lib.py. Photo use: none."""
from rules import r3_write_lib as L

vit = L.make(fold=("mlp",), tag="w_fold")
