"""Loop 2: GN per step (1..3, S=2) vs trained, all cached families; saves the GN neuron."""
from n7_lib import *
done = set(os.path.basename(p)[3:-4] for p in glob.glob(os.path.join(CD, "gn_*.npz")))
for key in (sys.argv[1:] or KEYS):
    if key in done or key.startswith("gn_"): continue
    f = load(key); o = {"children": len(f.cs), "kl_naive": float(f.k_naive.mean()), "trained4": round(f.share(f.L), 3)}; th = dict(f.parent); steps = []
    for it in range(3):
        th, h = gn(f, th, nit=1, lms=(1e-1, 1, 10, 100), als=(.5, 1.0), S=2); steps.append(round(f.share(th), 3))
    o["gn_steps"] = steps; np.savez(os.path.join(CD, f"gn_{key}.npz"), u=th["u"], b0=th["b0"], w=th["w"])
    print(key, json.dumps(o), f"{time.time() - t0:.0f}s", flush=True)
