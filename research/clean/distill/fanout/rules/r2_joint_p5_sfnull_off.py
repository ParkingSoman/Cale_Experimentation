"""JOINT variant: order=p5, stream fold=null (closed loop, regress teacher stream on kept student channels + 1), inner fold=off. See r2_joint_core.py. Photo use: none."""
from rules import r2_joint_core as J

r18 = J.make(order="p5", fold="off", sfold="null")
