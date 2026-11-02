"""D6 pictures of the two roles. Sheets (gallery.py: 8 strongest + 8 most negative spots per neuron):
  openers   = top-4 block-16 neurons by positive gate share, each followed by the block-17 reader it opens most
  inhibitors= top-4 by negative gate share, each followed by the reader it closes most
  volume    = top-6 block-16 neurons by volume share (and their gate share)"""
import os, numpy as np
from convnext import *
import gallery as G
m = load_convnext(); D = os.path.join(HF, "imagenet", "d5cache"); r = np.load(os.path.join(D, "d6_roles.npz")); vol, gate, en = r["vol"], r["gate"], r["en"]
b16, b17 = m.blocks[16], m.blocks[17]
W2 = np.array(b16.fc2.weight, np.float64) * np.array(b16.gamma, np.float64)[:, None]; Wc = W2 - W2.mean(0, keepdims=True)
U = np.array(b17.fc1.weight, np.float64) * np.array(b17.norm.weight, np.float64)[None]; imp17 = np.linalg.norm(np.array(b17.fc2.weight), axis=0)
E = U @ Wc                                          # (3072 readers, 3072 block-16 neurons): static edge
def reader(i, sign): s = sign * E[:, i] * imp17; return int(np.argmax(s))
rows = []
for i in np.argsort(-gate)[:4]:
    j = reader(i, +1); rows += [(16, int(i), f"OPENER 16/{i}: gate {gate[i]:+.3f}, volume {vol[i]:+.4f}"), (17, j, f"   reader 17/{j} (edge {E[j, i]:+.2f})")]
print(G.gallery(rows, "d/d6_openers.png", "block-16 gate OPENERS and the block-17 reader each opens most"))
rows = []
for i in np.argsort(gate)[:4]:
    j = reader(i, -1); rows += [(16, int(i), f"INHIBITOR 16/{i}: gate {gate[i]:+.3f}, volume {vol[i]:+.4f}"), (17, j, f"   reader 17/{j} (edge {E[j, i]:+.2f})")]
print(G.gallery(rows, "d/d6_inhibitors.png", "block-16 gate INHIBITORS and the block-17 reader each closes most"))
rows = [(16, int(i), f"VOLUME 16/{i}: volume {vol[i]:+.4f}, gate {gate[i]:+.3f}") for i in np.argsort(-vol)[:6]]
print(G.gallery(rows, "d/d6_volume.png", "block-16 VOLUME neurons (largest share of the stream size block 16 adds)"))
