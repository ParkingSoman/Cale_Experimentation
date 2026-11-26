"""STREAM round 3: rand null, second seed (seed 1), same form as r2_p6_rand_rho. Input X only; photos scoring only."""
from rules import r2_p6_core


def vit(T, f, X=None):
    return r2_p6_core.build(T, f, X, "rand", "rho", seed=1)
