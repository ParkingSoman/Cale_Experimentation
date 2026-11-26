"""Face-validity demo (not a pass/fail test): read 8 ImageNet head rows as read-like vectors on the final stream s4b1.
The head reads the pooled ReLU of the stream; reading its rows on the pre-ReLU cells is an approximation (pooling is
linear, the ReLU is not). Classes are drawn with a fixed seed, not picked. Class names appear only as the title of each
reading; they group or select nothing.
Output: results/demo_head.md
"""
import json
import os

import numpy as np

import vecread as vr

CLEAN = os.path.dirname(vr.HERE)


def main():
    A = np.load(os.path.join(CLEAN, "neuron", "results", "coarse", "r18", "w_A.npz"))
    ci = json.load(open(os.path.join(CLEAN, "neuron", "results", "read", "imagenet_class_index.json")))
    dic = vr.Dictionary("s4b1")
    held = vr.Held()
    classes = np.random.default_rng(2026).choice(1000, 8, replace=False)
    md = ["# Head rows read on the final stream (demo, approximate: ReLU ignored)", ""]
    for c in classes:
        r = vr.read_vector(dic, A["Wfc"][c], kind="read", held=held, top=5)
        md.append(vr.card_md(f"class {c}: {ci[str(c)][1]}", r).replace("# ", "## ", 1))
    open(os.path.join(vr.RES, "demo_head.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
