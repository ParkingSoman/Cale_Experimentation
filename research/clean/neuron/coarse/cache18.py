"""N11: cache ResNet-18's stage-4 input (stage-3 output, fp16, 256 x 14 x 14) and A's logits for each photo set,
plus the left-right flip of each train photo. Check: A from the cache reproduces A's logits."""
import os
import sys

import numpy as np
import torch

import r18

torch.set_grad_enabled(False)


def main():
    a = r18.load_a()
    ph = r18.net.Photos("read")
    sets = r18.photo_sets(ph.lab)
    sets["trainflip"] = sets["train"]
    for name in r18.SETS:
        ix = sets[name]
        f = os.path.join(r18.CACHE, f"{name}_x.npy")
        X = np.lib.format.open_memmap(f, "w+", np.float16, (len(ix), 256, 14, 14))
        Z = np.zeros((len(ix), 1000), np.float32)
        worst = 0.0
        for s in range(0, len(ix), 100):
            x = ph.tensor(ix[s:s + 100])
            if name == "trainflip":
                x = torch.flip(x, dims=[3])
            x = x.to(r18.DEV)
            h = a.layer3(a.layer2(a.layer1(r18.net.stem(a, x))))
            X[s:s + len(h)] = h.half().cpu().numpy()
            z = r18.net.head(a, a.layer4(h))
            Z[s:s + len(h)] = z.cpu().numpy()
            worst = max(worst, float((r18.run4(a, h.half().float()) - z).abs().max()))
        X.flush()
        del X
        np.save(os.path.join(r18.CACHE, f"{name}_logits.npy"), Z)
        np.save(os.path.join(r18.CACHE, f"{name}_ix.npy"), ix)
        print(f"{name}: {len(ix)} photos, fp16 cache vs fp32 logits max abs diff {worst:.4f}", flush=True)


if __name__ == "__main__":
    main()
