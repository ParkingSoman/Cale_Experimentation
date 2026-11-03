# serves: H-map
"""Write the class list for the blind reviewers."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from net import HERE, class_names  # noqa: E402

with open(os.path.join(HERE, "review", "classes.txt"), "w") as f:
    for i, n in enumerate(class_names()):
        f.write(f"{i} {n}\n")
