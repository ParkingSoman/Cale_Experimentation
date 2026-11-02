"""17 n2283: (a) class-subset test cars vs dogs vs birds vs rest; (b) essay check: is there a block 13-16 neuron on for
BOTH cars and dogs (shared parent before the car-vs-dog split)? On = image max > 0.5 x the neuron's 99.9th-percentile
max over the 20k analysis images. Lift = P(on | set) / P(on | all). Score = min(car lift, dog lift)."""
import numpy as np, json
from last_lib import *
ya = labels_ana(); nm = names()
CARS = [407, 436, 468, 511, 609, 627, 656, 661, 717, 751, 817, 581, 479]       # car classes (+ grille, car wheel)
DOGS = list(range(151, 269)); BIRDS = list(range(7, 25)) + list(range(80, 101)) + list(range(127, 147))
car, dog, bird = np.isin(ya, CARS), np.isin(ya, DOGS), np.isin(ya, BIRDS); rest = ~(car | dog | bird)
v, w = amax(17, 2283), amin(17, 2283)
print("(a) 17 n2283 mean image max:  cars %.2f  dogs %.2f  birds %.2f  rest %.2f  all %.2f" % (v[car].mean(), v[dog].mean(), v[bird].mean(), v[rest].mean(), v.mean()))
print("    mean image min:  cars %.2f  dogs %.2f  birds %.2f  rest %.2f" % (w[car].mean(), w[dog].mean(), w[bird].mean(), w[rest].mean()))
on = v > 0.5 * np.percentile(v, 99.9); print("    on-rate: cars %.3f dogs %.3f birds %.3f rest %.3f" % (on[car].mean(), on[dog].mean(), on[bird].mean(), on[rest].mean()))
d = cache(); out = []
for b in (13, 14, 15, 16):
    M = d[f"max{b}"][:].astype(np.float32); thr = 0.5 * np.percentile(M, 99.9, axis=0); ON = M > thr
    p = ON.mean(0); pc, pd, pb = ON[car].mean(0), ON[dog].mean(0), ON[bird].mean(0)
    ok = p > 0.002
    lc, ld, lb = np.where(ok, pc / np.maximum(p, 1e-9), 0), np.where(ok, pd / np.maximum(p, 1e-9), 0), np.where(ok, pb / np.maximum(p, 1e-9), 0)
    s = np.minimum(lc, ld)
    for n in np.argsort(-s)[:6]: out.append((float(s[n]), b, int(n), float(lc[n]), float(ld[n]), float(lb[n]), float(p[n]), float(pc[n]), float(pd[n])))
    # also: is the child's car-vs-dog split visible as a single neuron's two poles in this block? (cars on, dogs strongly off)
out.sort(reverse=True)
print("(b) top shared car+dog candidates: score=min(lift_car,lift_dog) | block n | lift car, dog, bird | P(on) all, cars, dogs")
for r in out[:12]: print("   %.2f | %d n%d | %.1f %.1f %.1f | %.3f %.3f %.3f" % r)
json.dump(out[:12], open("figs/t5_shared.json", "w"))
