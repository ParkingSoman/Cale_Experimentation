"""Big heat maps of one neuron on its top-12 and bottom-6 images (20k analysis set). Usage: python heat12.py B N"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import last_lib as L
b, n = int(sys.argv[1]), int(sys.argv[2])
L.OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
X, y, ana = L.data(); v = L.amax(b, n); u = L.amin(b, n)
top = ana[np.argsort(-v)[:12]]; bot = ana[np.argsort(u)[:6]]
print(L.heat_sheet(b, n, list(top), f"heat_b{b}_n{n}_top.png", f"b{b} n{n} top12"))
print(L.heat_sheet(b, n, list(bot), f"heat_b{b}_n{n}_neg.png", f"b{b} n{n} most negative"))
print("top classes", L.top_classes(v, 12)); print("neg classes", L.top_classes(u, 12, neg=True))
