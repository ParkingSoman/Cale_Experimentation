"""JOINT CLIP variant w_foldmlp. See r2_joint_vit.py. Photo use: none."""
from rules import r2_joint_vit as J

vit = J.make(mlp_a=0.0, fold=("mlp",))
