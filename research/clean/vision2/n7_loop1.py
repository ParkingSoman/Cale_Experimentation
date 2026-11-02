"""Loop 1: is training the ceiling, and does GN converge to it? (eval = held-out sealed images)"""
from n7_lib import *
fams = sys.argv[1:]
for key in fams:
    f = load(key); o = {"trained4": round(f.share(f.L), 3)}
    gtr = np.array(f.gix); L4g = train(f, 4, ix=gtr); o["trained4_gix"] = round(f.share(L4g), 3)
    L12 = train(f, 12, ix=gtr); o["trained12_gix"] = round(f.share(L12), 3)
    o["trainKL_v"] = {"cut": round(f.trainkl(f.parent) * 1e4, 2), "tr4": round(f.trainkl(L4g) * 1e4, 2), "tr12": round(f.trainkl(L12) * 1e4, 2)}
    th, hist = gn(f, dict(f.parent), nit=4, lms=(1e-1, 1, 10, 100), als=(.5, 1.0), verbose=False)
    o["gn4"] = round(f.share(th), 3); o["gn_hist"] = hist; o["trainKL_v"]["gn4"] = round(f.trainkl(th) * 1e4, 2)
    print(key, json.dumps(o), f"{time.time() - t0:.0f}s", flush=True)
