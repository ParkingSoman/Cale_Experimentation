"""Generic input: images that come from no data set (FANOUT plan 2026-11-26, Shonu confirmed generic input is allowed).

pink(n): Gaussian noise with a natural-image amplitude spectrum (amplitude ~ 1 / f^alpha, alpha = 1).
The three colour channels are independent (no colour statistics, because those would come from data).
Each channel is scaled to mean 0, std 1 per image. That is the range the network's own input normalization targets.
This is the only input a calculation may run. Photos are for scoring only.
"""
import torch


def pink(n, size=224, alpha=1.0, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, 3, size, size, generator=g)
    fy = torch.fft.fftfreq(size).view(-1, 1)
    fx = torch.fft.rfftfreq(size).view(1, -1)
    f = torch.sqrt(fy ** 2 + fx ** 2)
    f[0, 0] = 1.0
    amp = 1.0 / f ** alpha
    amp[0, 0] = 0.0
    x = torch.fft.irfft2(torch.fft.rfft2(w) * amp, s=(size, size))
    x = (x - x.mean((2, 3), keepdim=True)) / x.std((2, 3), keepdim=True)
    return x.to(device)
