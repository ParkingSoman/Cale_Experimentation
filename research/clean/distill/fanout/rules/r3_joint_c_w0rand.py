"""JOINT r3 CLIP: plain random-MLP-order NULL for P5 w0 (stream floor coords, heads floor act, MLP seeded random order, no fold). Photo use: none."""
from rules import r3_joint_clip as J

vit = J.make(stream="floor", cx="x", heads="act", mlp="rand", fold=False)
