"""JOINT variant: order=p5, stream fold=off, inner fold=aff. See r2_joint_core.py. Photo use: none."""
from rules import r2_joint_core as J

r18 = J.make(order="p5", fold="aff", sfold="off")
