"""Recipe parents v2 (family "recipe"): same as recipe_v1, but families are SEEDED by the cut's top-K neurons.
Each other candidate pole joins the seed with the most similar recipe (cosine >= 0.2), else it is dropped. Seeds with
no members stay as-is (cut). Diagnosis behind it: results/clean-k-diag-v1.json (v1 lost on WHICH units got a slot).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recipe_v1  # noqa: E402

METHOD = {"name": "recipe_v2", "family": "recipe", "minutes": 6.0}


def build(model, tok, L, K, cal, log=print):
    return recipe_v1.build(model, tok, L, K, cal, log=log, family_mode="seeded", seed_cos=0.2)
