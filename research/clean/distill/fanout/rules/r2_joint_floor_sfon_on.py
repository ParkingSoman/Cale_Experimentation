"""JOINT variant: order=floor, stream fold=on (closed loop, regress teacher stream on kept student channels + 1), inner fold=on. See r2_joint_core.py. Photo use: none."""
from rules import r2_joint_core as J

r18 = J.make(order="floor", fold="on", sfold="on")
