"""N11 setup check on ResNet-18 (torchvision IMAGENET1K_V1): where is the car family split? On car reading photos,
replace the residual branch r of a set of blocks by its mean over car photos (per channel and position), and measure
in-car accuracy, car mass and in-car KL as in where_cars.py."""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "curves", "map_agents", "resnet18"))
import net
torch.set_grad_enabled(False)
CARS = [407, 436, 468, 511, 609, 627, 656, 661, 717, 751, 817]
BLOCKS = [(s, b) for s in (1, 2, 3, 4) for b in (0, 1)]

def run(m, x, fill):
    h = net.stem(m, x)
    for s, lay in net.stages(m).items():
        for b, bl in enumerate(lay):
            inner = torch.relu(bl.bn1(bl.conv1(h))); r = bl.bn2(bl.conv2(inner))
            if (s, b) in fill: r = fill[(s, b)].expand_as(r)
            skip = h if bl.downsample is None else bl.downsample(h)
            h = torch.relu(r + skip)
    return net.head(m, h)

def main():
    m = net.load(); ph = net.Photos("read")
    idx = np.where(np.isin(ph.lab, CARS))[0]; Y = ph.lab[idx]
    X = torch.cat([ph.tensor(idx[a:a + 64]) for a in range(0, len(idx), 64)])
    rs = {}
    h = None
    sums = {k: 0 for k in BLOCKS}
    for a in range(0, len(X), 64):
        x = X[a:a + 64]; h = net.stem(m, x)
        for s, lay in net.stages(m).items():
            for b, bl in enumerate(lay):
                p = net.block_parts(bl, h); sums[(s, b)] = sums[(s, b)] + p["r"].sum(0); h = p["h"]
    mean = {k: (v / len(X)).unsqueeze(0) for k, v in sums.items()}
    ci = torch.as_tensor(CARS); yi = torch.as_tensor([CARS.index(int(y)) for y in Y])
    Z = torch.cat([run(m, X[a:a + 64], {}) for a in range(0, len(X), 64)])
    def score(ks):
        out = torch.cat([run(m, X[a:a + 64], {k: mean[k] for k in ks}) for a in range(0, len(X), 64)])
        p = F.softmax(out, 1)
        lq = F.log_softmax(out[:, ci], 1); lp = F.log_softmax(Z[:, ci], 1)
        return dict(in_car_acc=float((p[:, ci].argmax(1) == yi).float().mean()), car_mass=float(p[:, ci].sum(1).mean()),
                    in_car_kl=float((lp.exp() * (lp - lq)).sum(1).mean()))
    res = {"photos": len(X), "A": score([])}
    print(len(X), "car photos. ResNet-18:", res["A"])
    sets = [[k] for k in BLOCKS] + [BLOCKS[i:] for i in range(1, 8)]
    for ks in sets:
        r = score(ks); name = "+".join(f"s{s}b{b}" for s, b in ks); res[name] = r
        print(f"{name:40s} in-car acc {r['in_car_acc']:.3f}  car mass {r['car_mass']:.3f}  in-car KL {r['in_car_kl']:.3f}")
    json.dump(res, open(os.path.join(HERE, "..", "results", "coarse", "where_cars18.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
