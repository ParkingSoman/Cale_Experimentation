"""JOINT CLIP variant foldboth. See r2_joint_vit.py. Photo use: none."""
from rules import r2_joint_vit as J

vit = J.make(mlp_a=1.0, fold=("heads", "mlp"))
