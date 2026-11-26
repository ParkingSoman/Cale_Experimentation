"""JOINT r3 CLIP: r3_joint_c_on with heads ranked by WEIGHT (magcut) instead of floor activity (LOO of the head order). Photo use: none."""
from rules import r3_joint_clip as J

vit = J.make(stream="rG", cx="x", heads="w", mlp="w", fold=True)
