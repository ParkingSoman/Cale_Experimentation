"""N11 setup check: where does A split the car family? On car photos only, replace the MLP output of a window of
blocks by its mean over car photos (per channel and position), and measure:
  in-car accuracy  = argmax over the 11 car classes equals the label (A's own value is the reference)
  car mass         = probability on the 11 car classes (does the network still say "some car")
  in-car KL        = KL of A's distribution restricted to the 11 cars (renormalized) to the network's.
Photos: every cached photo (train, held, cal, test) whose label is a car. Choosing only, no scoring set.
"""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
import cc
C = cc.C
torch.set_grad_enabled(False)
CARS = [407, 436, 468, 511, 609, 627, 656, 661, 717, 751, 817]  # ambulance, beach wagon, cab, convertible, jeep, limousine, minivan, Model T, pickup, racer, sports car

def car_x3():
    lab = C.net.Photos("read").lab
    xs, zs, ys = [], [], []
    for name in ("train", "held", "cal", "test"):
        c = C.Cache(name); r = np.where(np.isin(lab[c.ix], CARS))[0]
        x, z = c.get(r); xs.append(x.half()); zs.append(z); ys.append(torch.as_tensor(lab[c.ix[r]]))
    return torch.cat(xs), torch.cat(zs), torch.cat(ys)

def main():
    base = C.load("base").eval(); BL = C.blocks(base)
    X, Z, Y = car_x3(); n = len(X)
    # mean MLP output of each block on car photos (per channel and position)
    store = {}
    hs = [b.block[5].register_forward_hook(lambda mod, i, o, g=g: store.setdefault(g, []).append(o.sum(0).float().cpu().double())) for g, b in enumerate(BL)]
    for a in range(0, n, 50): C.tail(base, X[a:a + 50].float())
    for h in hs: h.remove()
    mean = {g: (sum(v) / n).float().to(C.DEV) for g, v in store.items()}
    ci = torch.as_tensor(CARS, device=C.DEV); yi = torch.as_tensor([CARS.index(int(y)) for y in Y])
    def score(window):
        hs = [BL[g].block[5].register_forward_hook(lambda mod, i, o, g=g: mean[g].expand_as(o)) for g in window]
        out = torch.cat([C.tail(base, X[a:a + 50].float()) for a in range(0, n, 50)])
        for h in hs: h.remove()
        p = F.softmax(out.float(), 1); pz = F.softmax(Z.float(), 1)
        acc = float((p[:, ci].argmax(1).cpu() == yi).float().mean())
        mass = float(p[:, ci].sum(1).mean())
        lq = F.log_softmax(out.float()[:, ci], 1); lp = F.log_softmax(Z.float()[:, ci], 1)
        kl = float((lp.exp() * (lp - lq)).sum(1).mean())
        return dict(in_car_acc=acc, car_mass=mass, in_car_kl=kl)
    res = {"photos": n, "A": score([])}
    print(f"{n} car photos. A: {res['A']}", flush=True)
    wins = [list(range(a, a + 3)) for a in range(0, 28, 3)] + [list(range(a, a + 6)) for a in range(0, 25, 3)] + [list(range(a, 30)) for a in (6, 12, 15, 18, 21, 24, 27)]
    for w in wins:
        r = score(w); res[f"{w[0]}-{w[-1]}"] = r
        print(f"blocks {w[0]:2d}-{w[-1]:2d}: in-car acc {r['in_car_acc']:.3f}  car mass {r['car_mass']:.3f}  in-car KL {r['in_car_kl']:.3f}", flush=True)
    json.dump(res, open(os.path.join(cc.OUT, "where_cars.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
