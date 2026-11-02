# serves: T9
# serves: T11
"""Train the toy teachers of COARSE-THEORY.md Part C (toy chains). Not a check: it makes the objects the checks use.
Usage: train_teacher.py <variant: gated | relu1 | selfgate> <sym: 1 | 0>
"""
import sys, time, torch
import chain as C

variant, sym = sys.argv[1], sys.argv[2] == "1"
torch.manual_seed(0)
X, y = C.sample(3000 if sym else 24000, seed=1, sym=sym)
Xv, yv = C.sample(500, seed=2, sym=True)
T = C.Teacher(seed=0, variant=variant)
opt = torch.optim.Adam(T.parameters(), 1e-3 if variant == "selfgate" else 3e-3)
t0 = time.time()
for it in range(3000 if variant == "selfgate" else 1500):
    idx = torch.randint(len(X), (256,))
    loss = torch.nn.functional.cross_entropy(T(X[idx]), y[idx])
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 300 == 0 or it == (2999 if variant == "selfgate" else 1499):
        with torch.no_grad():
            acc = (T(Xv).argmax(1) == yv).double().mean().item()
        print(f"it {it} loss {loss.item():.3f} val acc {acc:.3f} t {time.time() - t0:.0f}s", flush=True)
torch.save(T.state_dict(), f"teacher_{variant}_{'sym' if sym else 'asym'}.pt")
