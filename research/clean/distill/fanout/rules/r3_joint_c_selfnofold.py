"""JOINT r3 CLIP variant: stream="rG", cx="self", heads="act", mlp="w", fold=False. See r3_joint_clip.py. Photo use: none."""
from rules import r3_joint_clip as J

vit = J.make(stream="rG", cx="self", heads="act", mlp="w", fold=False)
