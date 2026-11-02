"""Loop 3: converge GN on stage-4 families (S=4, up to 8 steps, step accepted only if held-out-train KL drops > 1%)."""
from n7_lib import *
for key in [k for k in KEYS if int(k.split("_")[0]) >= 15]:
    f = load(key); th = dict(f.parent); kv = f.trainkl(th); steps = []
    for it in range(8):
        th2, h = gn(f, th, nit=1, lms=(1e-1, 1, 10, 100), als=(.5, 1.0), S=4); kv2 = f.trainkl(th2)
        if kv2 > kv * .99: break
        th, kv = th2, kv2; steps.append(round(f.share(th), 3))
    np.savez(os.path.join(CD, f"gn_{key}.npz"), u=th["u"], b0=th["b0"], w=th["w"])
    print(key, "trained4 %.3f" % f.share(f.L), "GN steps", steps, f"{time.time() - t0:.0f}s", flush=True)
