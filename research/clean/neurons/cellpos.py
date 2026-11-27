# scale: 3 named layer4.0 channels; net=A, W50R_s0, page-v2 50% build; photos=10,000 val (read set); where=mac
"""Where in the 7 x 7 grid each neuron fires: mean pre-activation per cell position, fire rate per position, and the
share of each photo's max cell that sits on the outer ring (24 of 49 cells = .49 if position does not matter)."""
import os, sys
import numpy as np, torch
CLEAN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("distill", "distill/tree", "distill/fanout"):
    sys.path.insert(0, os.path.join(CLEAN, p))
import build as TB, board, wide18  # noqa: E402
torch.set_grad_enabled(False)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PX = np.load(os.path.join(CLEAN, "hf/imagenet/val_224.npy"), mmap_mode="r"); READ = np.arange(0, 50000, 5)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
A = board.r18_A().cpu().eval()
d = torch.load(os.path.join(CLEAN, "distill/tree/out/cache_r18_724b03416e.pt"), weights_only=False)
TB.COARSE_PARENT = False; TB.WRITE_PARTNER = False
S2 = TB.build_r18(A, 0.5, None, st=d["st"], tr=d["tr"], verbose=False).eval()
dd = torch.load(os.path.join(CLEAN, "neuron/results/coarse/r18/distill_whole/W50R_s0.pt"), map_location="cpu", weights_only=False)
W = wide18.Student(tuple(dd["widths"])); W.load_state_dict(dd["state"]); W.eval()
ring = torch.zeros(7, 7, dtype=torch.bool); ring[0] = ring[-1] = True; ring[:, 0] = ring[:, -1] = True; ring = ring.reshape(-1)
for name, m, ch in (("X", A, 184), ("Y", W, 139), ("Z", S2, 93)):
    m.to(DEV); M = []
    for a in range(0, len(READ), 250):
        x = ((torch.from_numpy(np.asarray(PX[READ[a:a + 250]])).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD).to(DEV)
        s3 = m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x)))))))
        M.append(m.layer4[0].bn1(m.layer4[0].conv1(s3))[:, ch].flatten(1).cpu())
    m.cpu(); M = torch.cat(M)
    mx = M.argmax(1); top = M.max(1).values.argsort(descending=True)[:250]
    print(f"\n{name}: share of max cells on the outer ring: all photos {float(ring[mx].float().mean()):.2f}, top-250 photos {float(ring[mx[top]].float().mean()):.2f} (no position effect = .49)")
    print(" mean pre-activation per cell position (rows top->bottom):")
    for r in M.mean(0).reshape(7, 7): print("  " + " ".join(f"{float(v):+.2f}" for v in r))
    print(" fire rate per cell position:")
    for r in (M > 0).float().mean(0).reshape(7, 7): print("  " + " ".join(f"{float(v):.2f}" for v in r))
