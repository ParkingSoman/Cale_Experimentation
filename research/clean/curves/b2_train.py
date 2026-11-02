"""Bench B2 ruler: coarse read, write AND reader biases trained (NEURON-MATH Part II). Warm start from a B1 checkpoint."""
import sys, os, json, numpy as np, torch, torch.nn as nn
from inception import *
from family import Family, Data, train, evaluate, IN4A

class FamilyRB(Family):
    """Family plus trainable reader-bias changes (raw units = 0.1 x parameter; 4a biases are ~.26 +- .09)."""
    def add_rb(self, m):
        self.cbp = nn.ParameterList([nn.Parameter(torch.zeros(getattr(m, n).weight.shape[0])) for n in IN4A])
    @property
    def colbias(self):
        return [p * 0.1 for p in self.cbp]

if __name__ == "__main__":
    src, tag = sys.argv[1], sys.argv[2]; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    m = load_model(); D = Data(seed=0, ndev=1000); fam = CURVES
    ck = torch.load(f"cache/ckpt/{src}.pt", map_location="cpu", weights_only=False)
    fm = FamilyRB(m, ck["fam"], ck["k"], ks=ck.get("ks", 5)); fm.load_state_dict(ck["state"]); fm.add_rb(m); fm = fm.to(DEV)
    print("start dev-1000", evaluate(m, fm, D, D.dv), flush=True)
    best, step, hist = train(m, fm, D, lr=1e-2, steps=steps, eval_every=125, patience=4)
    torch.save(dict(state=fm.state_dict(), fam=ck["fam"], k=ck["k"], ks=ck.get("ks", 5), src=src, rb=True), f"cache/ckpt/{tag}.pt")
    print("BEST", tag, best, "at step", step, flush=True)
