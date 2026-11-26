"""JOINT variant: order=p5null, mean=mean, fold=off (ResNet-18). See r2_joint_core.py. Photo use: none."""
from rules import r2_joint_core as J

r18 = J.make(order="p5null", mean="mean", fold="off")
