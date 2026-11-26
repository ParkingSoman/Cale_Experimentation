"""P4 round 2: OV merge NULL (same form): each cut head merges into a RANDOM kept head. See r2_p4_ovmerge.py."""
from rules import r2_p4_ovmerge as L


def vit(T, f, X=None):
    return L.build(T, f, X, null="rand", tag="ovmerge_null")
