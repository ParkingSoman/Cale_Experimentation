"""JOINT r3 CLIP: r3_joint_c_on + readout fold (post). Photo use: none."""
from rules import r3_joint_clip as J

vit = J.make(stream="rG", cx="x", heads="act", mlp="w", fold=True, post=True)
