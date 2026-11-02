from net import train
for name, C, H, seed in (("S", 32, 128, 0), ("L", 128, 512, 0), ("LN", 128, 128, 0), ("L2", 128, 512, 1)):
    train(name, C, H, 25, init_seed=seed)
