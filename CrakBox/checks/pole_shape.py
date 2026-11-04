"""How common is half of the largest firing on each pole? Reading photos only. A sheet's fire items are a sample of firings at or above half of the largest, so this says what those items are."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from networks import get
net = get("resnet18")
nodes = json.load(open(os.path.join(ROOT, "checks", "smoke_scope.json")))["nodes"] + ["s3b1:11", "s3b1i:40", "s2b1:9", "s2b0i:20", "s1b1:7", "s1b0i:3"]
print("node | pos: share of photos at >= .5, value at the 99th percentile / largest | neg: same")
ratios = []
for n in nodes:
    row = [n]
    for pole in net.poles(n):
        v, _ = net.peaks(n, pole, "read")
        big = v.max()
        row += [round(float((v >= .5 * big).mean()), 4), round(float(np.percentile(v, 99) / big), 3)]
    print(*row)
