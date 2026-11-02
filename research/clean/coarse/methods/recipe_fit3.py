"""Recipe + short fit (family "recipe-fit"): recipe_v3p12 gives the gates (36 strong neurons as-is + 12 recipe parents
for tail families). Gates are FROZEN; values and writes get the same 3-minute window fit as shortfit3.
Interp supplies WHAT (base-only coarse families), the fit supplies HOW."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recipe_v1  # noqa: E402
from _fit import fit_window  # noqa: E402

METHOD = {"name": "recipe_fit3", "family": "recipe-fit", "minutes": 4.5}


def build(model, tok, L, K, cal, log=print):
    layer = recipe_v1.build(model, tok, L, K, cal, log=log, family_mode="tail", n_parents=12)
    layer, steps = fit_window(model, L, layer, cal, minutes=3.0, freeze=("G",))
    log(f"  recipe_fit3 L{L}: {steps} fit steps")
    return layer
