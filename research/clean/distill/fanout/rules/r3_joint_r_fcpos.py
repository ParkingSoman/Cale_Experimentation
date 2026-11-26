"""JOINT r3 ResNet: floor order + stream fold + inner fold, fc fold = pos. See r3_joint_r18core.py. Photo use: none."""
from rules import r3_joint_r18core as J

r18 = J.make(order="floor", fold="on", sfold="on", fc="pos")
