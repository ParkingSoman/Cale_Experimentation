"""Recipe parents v3 (family "recipe"): the cut's top K-12 neurons stay as-is; 12 recipe parents stand in for
families of the remaining strong poles (the long tail). Mirrors the toy (parents replace removed children, strong
neurons stay). Diagnosis behind it: v2 blurred important seed neurons (families up to 37 at L15).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recipe_v1  # noqa: E402

METHOD = {"name": "recipe_v3p12", "family": "recipe", "minutes": 6.0}


def build(model, tok, L, K, cal, log=print):
    return recipe_v1.build(model, tok, L, K, cal, log=log, family_mode="tail", n_parents=12)
