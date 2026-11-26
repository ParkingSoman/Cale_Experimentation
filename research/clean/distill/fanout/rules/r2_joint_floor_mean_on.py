"""JOINT variant: order=floor, mean=mean, fold=on (ResNet-18). See r2_joint_core.py. Photo use: none."""
from rules import r2_joint_core as J

r18 = J.make(order="floor", mean="mean", fold="on")
