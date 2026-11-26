"""JOINT r3 ResNet: same-form NULL of fc=pos: stream fold null (shuffled pairing) everywhere, fc per position. Photo use: none."""
from rules import r3_joint_r18core as J

r18 = J.make(order="floor", fold="on", sfold="null", fc="pos")
