"""F2' (Statement 25.6): Fisher kept exact on the top-K reader modes (u_k of the G-factor) x all 962 inputs, plain KFC on
the complement; control: exact on all readers x top-K activity modes (v_n of the A-factor)."""
import numpy as np, torch, time
src = open("c17_fisher.py").read(); exec(src.split("G = torch.zeros(NRD, NRD")[0])
C = torch.load("cache/c17_fisher.pt"); Gf, Af = C["G"].to(DEV), C["A"].to(DEV)
UG, UA = C["UG"].to(DEV).flip(1), C["UA"].to(DEV).flip(1)                               # descending eigenvalues
Ks = [5, 10, 20, 40, 80]; res = {(side, K): {k: 0.0 for k in Vs} for side in "ra" for K in Ks}
def proj(V, side, K):
    if side == "r": U = UG[:, :K]; return U @ (U.T @ V)
    U = UA[:, :K]; return (V @ U) @ U.T
for i in range(0, 500, 10):
    Gm, _, _ = gammas(TR[i:i + 10])
    for side in "ra":
        for K in Ks:
            for k, V in Vs.items():
                res[(side, K)][k] += float(((Gm * proj(V, side, K)).sum((2, 3)) ** 2).sum()) / 2
for side in "ra":
    for K in Ks:
        L = {}
        for k, V in Vs.items():
            Vp = proj(V, side, K); Vc = V - Vp
            L[k] = res[(side, K)][k] / 500 + 0.5 * float((Vc * (Gf @ Vc @ Af)).sum())
        print(f"{'reader' if side == 'r' else 'activity'} modes K={K:3d}: S {L['S']:.4f}  S+rem/S {L['S+rem'] / L['S']:.3f}  T/S {L['T'] / L['S']:.3f}  4d/S {L['4d'] / L['S']:.3f}", flush=True)
