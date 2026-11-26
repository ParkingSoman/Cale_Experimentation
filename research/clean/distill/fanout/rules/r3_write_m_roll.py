"""WRITE r3 variant m_roll. See rules/r3_write_lib.py. Photo use: none."""
from rules import r3_write_lib as L

vit = L.make(cut=("mlp",), fold=("mlp",), null="roll", tag="m_roll")
