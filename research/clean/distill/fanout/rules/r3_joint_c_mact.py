"""JOINT r3 CLIP variant: stream="rG", cx="x", heads="act", mlp="act", fold=True. See r3_joint_clip.py. Photo use: none."""
from rules import r3_joint_clip as J

vit = J.make(stream="rG", cx="x", heads="act", mlp="act", fold=True)
