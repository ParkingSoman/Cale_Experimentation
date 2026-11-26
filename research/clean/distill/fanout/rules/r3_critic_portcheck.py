"""CRITIC round 3: port check of JOINT's CLIP fold (rules/r2_joint_vit.build) on an MLP-ONLY cut.

Object (picked by): pr_w of every block; kept MLP units by weight (P5 a = 0, same as r2_joint_v_w); stream and heads
kept at FULL width. Same build code as r2_joint_v_w_foldmlp, only keep_sets is changed.
Question: does the port's fold reproduce P4's MLP-only fold (r2_p4_wfold_mlp)? If yes, the whole-cut harm comes from
the fold meeting the stream/head cut, not from a port bug.
Photo use: none. Weights + the generic input X only.
"""
import torch

import vitgen
from rules import r2_joint_vit as J


def vit(T, f, X=None):
    orig = J.keep_sets

    def ks(T_, f_, X_, mlp_a=1.0, head_a=1.0):
        sd, hk, mk = orig(T_, f_, X_, mlp_a, head_a)
        D, H, _ = vitgen.sizes(1.0, T_)
        return torch.arange(D), [torch.arange(H) for _ in hk], mk

    J.keep_sets = ks
    try:
        return J.build(T, f, X, mlp_a=0.0, fold=("mlp",))
    finally:
        J.keep_sets = orig
