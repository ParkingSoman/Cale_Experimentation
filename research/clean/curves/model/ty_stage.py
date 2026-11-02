# serves: TY5
"""Checks of Part B.Y.2 (COARSE-THEORY.md): Theorem TY4 (bookkeeping with skips and positions), Proposition TY5.
  A  TY4: a two-layer network with a skip from the input into layer 2 and two positions, a planted tree map (car gate
     -> taxi / ambulance; dog). Every claim TH1 uses has error 0. For random boxes of the taxi group (k = 1, weights
     shared over positions): kept channels keep N's values, the box reads N's z_T, the box is constant off the taxi
     area, and TH1's assembly gives y = V C + U h_J + t on taxi and y_N elsewhere
  B  the taxi group's local least for k = 1 by a local optimiser (reported; a build's D_P is at least its taxi terms,
     and the assembly of A reaches them, so this is the instance's V_b up to the optimiser)
  A2 DY3 item 2: an average-pool copy across a 2-to-1 tiling keeps the pooled value, so the judgment shift survives;
     a stride copy (one position per tile) does not (the referee's counterexample h_J = (2, 0))
  C  TY5: the identity of part 1; part 3's numbers; k = 0 least = log of the mean softmax (no optimiser beats it);
     the D2 choice's excess = the KL of part 2; part 4's bounds on random builds
Run: python model/ty_stage.py"""
import numpy as np

rng = np.random.default_rng(3110)
fails, COUNT = [], {}


def check(name, ok, info=""):
    COUNT[name.split()[0]] = COUNT.get(name.split()[0], 0) + 1
    if not ok:
        fails.append(f"{name} {info}")


relu = lambda t: np.maximum(t, 0)


def softmax(y):
    e = np.exp(y - y.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def kl(p, q):
    return (p * (np.log(p) - np.log(q))).sum(-1)


# ---------------- A, B: the planted network with a skip and two positions ----------------
T = 2
pts = []   # (area, input array (T, 8))
for _ in range(6):
    x = np.full((T, 8), -1.0); x[:, 0] = 1.0
    x[:, 2:4] = rng.uniform(0.5, 3, size=(T, 2)); x[:, 4:6] = -10.5; x[:, 6] = 1.0
    pts.append(("taxi", x))
for _ in range(6):
    x = np.full((T, 8), -1.0); x[:, 0] = 1.0
    x[:, 2:4] = -10.5; x[:, 4:6] = rng.uniform(0.5, 3, size=(T, 2)); x[:, 7] = 1.0
    pts.append(("amb", x))
for _ in range(4):
    x = np.full((T, 8), -1.0); x[:, 0] = -1.0; x[:, 1] = rng.uniform(0.5, 2, size=T)
    x[:, 2:6] = -10.0
    pts.append(("dog", x))
w_pts = np.full(len(pts), 1.0 / len(pts))
Wout = rng.normal(size=(3, 6))       # logits read pooled (T1, T2, A1, A2, Jt+Ja+DD as 3 more)
cN = rng.normal(size=3)


def layer1(x):
    G = relu(x[:, 0])                 # car gate (district car, region taxi + amb)
    D1 = relu(x[:, 1])                # dog district
    return G, D1


def layer2_pre(x, G, D1):
    """pre-activations of layer 2; each reads layer 1 AND the input (the skip)."""
    zT = np.stack([0.5 * G + x[:, 2], 0.5 * G + x[:, 3]], 1)          # taxi channels T1, T2
    zA = np.stack([0.5 * G + x[:, 4], 0.5 * G + x[:, 5]], 1)          # ambulance channels A1, A2
    zJ = np.stack([x[:, 6], x[:, 7]], 1)                              # judgments Jt, Ja
    zD = (D1 + x[:, 1])[:, None]                                      # dog stream channel
    return zT, zA, zJ, zD


def N_forward(x):
    G, D1 = layer1(x)
    zT, zA, zJ, zD = layer2_pre(x, G, D1)
    H = np.concatenate([relu(zT), relu(zA), relu(zJ), relu(zD)], 1)   # (T, 7): T1 T2 A1 A2 Jt Ja DD
    p = H.mean(0)
    y = Wout @ np.concatenate([p[:4], [p[4] + p[5] + p[6]], [0.0]]) + cN
    return dict(G=G, D1=D1, zT=zT, H=H, p=p, y=y)


NV = [N_forward(x) for _, x in pts]
yN = np.array([v["y"] for v in NV])
# claims at error 0
for (ar, x), v in zip(pts, NV):
    on = lambda a: np.abs(a).sum() > 0
    check("A-dist", (on(v["H"][:, 0:2]) or True) and (not on(v["H"][:, 0:2]) or ar == "taxi"))
    check("A-dist", not on(v["H"][:, 2:4]) or ar == "amb")
    check("A-dist", not on(v["H"][:, 6]) or ar == "dog")
    check("A-dist", not on(v["G"]) or ar in ("taxi", "amb"))
    if ar == "taxi":
        check("A-judg", abs(v["p"][4] - 1) < 1e-12)
    else:
        check("A-judg", abs(v["p"][4]) < 1e-12)
        check("A-shel", np.abs(v["zT"] + 10.0).max() < 1e-12, str(v["zT"]))
    # tree map nesting: layer-2 regions inside layer-1 regions
    if ar == "taxi" or ar == "amb":
        check("A-nest", on(v["G"]))

# random boxes for the taxi group S = {T1, T2}, k = 1, J_w = {Jt}
taxi = np.array([a == "taxi" for a, _ in pts])
for trial in range(50):
    m, beta = rng.normal(size=2), rng.normal()
    Cbox = np.array([relu(v["zT"] @ m + beta).mean() for v in NV])     # box pooled value (shared over positions)
    check("A-const", np.ptp(Cbox[~taxi]) < 1e-12)
    # TH1 assembly: target per-area logits V C + U h_J + t on taxi, N elsewhere
    V, U, t = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
    vS = V * Cbox[~taxi][0]
    # build: kept channels (A1, A2, Ja, DD) keep N's logit weights; Jt gets U + s a^T with a = 1, s = t - cN + vS
    s = t - cN + vS
    cB = cN - vS
    yB = []
    for v, c in zip(NV, Cbox):
        p = v["p"]
        kept = Wout @ np.concatenate([[0, 0], p[2:4], [p[5] + p[6]], [0.0]])
        yB.append(kept + V * c + (U + s) * p[4] - Wout[:, 4] * 0 + cB)
    yB = np.array(yB)
    # N's own logit weight on Jt is Wout[:, 4]; the build replaces it, so compare on each area as TH1 states:
    target_taxi = np.array([V * c + U * v["p"][4] + t for v, c in zip(NV, Cbox)])
    # on taxi the kept channels are off except Jt, so y_B = V C + U h_J + t
    check("A-taxi", np.abs(yB[taxi] - target_taxi[taxi]).max() < 1e-12)
    # elsewhere: y_B = y_N minus N's Jt term (Jt off there) -> equal to y_N
    check("A-else", np.abs(yB[~taxi] - yN[~taxi]).max() < 1e-12, f"{np.abs(yB[~taxi] - yN[~taxi]).max():.2e}")

# B: least of the whole build vs the taxi group's local least (k = 1), by many random starts of a local optimiser
def taxi_terms(params):
    m, beta, V, U, t = params[:2], params[2], params[3:6], params[6:9], params[9:12]
    C = np.array([relu(v["zT"] @ m + beta).mean() for v in NV])
    y = np.array([V * c + U * v["p"][4] + t for v, c in zip(NV, C)])
    return (w_pts[taxi] * kl(softmax(yN[taxi]), softmax(y[taxi]))).sum()


def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


best_local = np.inf
for start in range(8):
    x = rng.normal(size=12)
    for it in range(300):
        x -= 0.5 * num_grad(taxi_terms, x)
    best_local = min(best_local, taxi_terms(x))
# any whole build is >= its taxi terms (other areas >= 0); the assembled build reaches the taxi terms exactly (A-else)
check("B-lower", best_local >= -1e-12)
print(f"B: least taxi terms found (k = 1): {best_local:.5f}")

# A2: copies across a change of positions
for trial in range(200):
    hv = relu(rng.normal(size=4))                     # a channel at 4 positions, tiled into 2 tiles of 2
    avg = relu(hv.reshape(2, 2).mean(1))              # average-pool copy
    check("A2-avg", abs(avg.mean() - hv.mean()) < 1e-12)
hJ = np.array([2.0, 0.0])                             # judgment value pooled = 1 on the area
check("A2-avgJ", abs(relu(hJ.reshape(1, 2).mean(1)).mean() - 1) < 1e-12)
check("A2-strideJ-fails", abs(hJ[1:].mean() - 1) > 0.5)   # a stride copy taking position 2 carries 0

# ---------------- C: Proposition TY5 ----------------
for trial in range(200):
    n, ny = rng.integers(2, 7), rng.integers(2, 6)
    w = rng.uniform(0.1, 1, size=n)
    P = softmax(rng.normal(size=(n, ny)) * 2)
    q = softmax(rng.normal(size=ny))
    mA = w.sum(); pbar = (w[:, None] * P).sum(0) / mA
    lhs = (w * kl(P, q)).sum()
    rhs = (w * kl(P, pbar[None, :])).sum() + mA * kl(pbar, q)
    check("C-ident", abs(lhs - rhs) < 1e-12, f"{lhs - rhs:.2e}")
    # k = 0: KL-least constant = log pbar; no optimiser beats it
    Y = rng.normal(size=(n, ny)) * 2
    P = softmax(Y); pbar = (w[:, None] * P).sum(0) / mA
    f = lambda t: (w * kl(P, softmax(np.broadcast_to(t, (n, ny))))).sum()
    t0 = np.log(pbar)
    t = rng.normal(size=ny)
    for it in range(500):
        t -= 0.5 * ((w[:, None] * (softmax(np.broadcast_to(t, (n, ny))) - P)).sum(0))
    check("C-k0", f(t) >= f(t0) - 1e-10, f"{f(t) - f(t0):.2e}")
    # D2 choice: centred mean of the logits; excess = mA KL(pbar || softmax(ybar))
    Pp = np.eye(ny) - 1.0 / ny
    ybar = (w[:, None] * (Y @ Pp)).sum(0) / mA
    check("C-excess", abs((f(ybar) - f(t0)) - mA * kl(pbar, softmax(ybar))) < 1e-10)
    # part 4: random builds
    YB = Y + rng.normal(size=(n, ny)) * rng.uniform(0.1, 3)
    ts = np.linspace(0, 1, 201)
    mu = min(softmax(Y + tt * (YB - Y)).min() for tt in ts)
    D = (w * kl(P, softmax(YB))).sum()
    D2 = 0.5 * (w * (((YB - Y) @ Pp) ** 2).sum(1)).sum()
    check("C-lowerF", D >= mu * D2 * (1 - 1e-3) - 1e-12, f"{D:.3e} {mu * D2:.3e}")
    check("C-upperF11", D <= D2 / 2 + 1e-12)

# part 3's numbers
s3 = 2 * np.log(3)
Ytoy = np.array([[s3, 0.0], [0.0, 0.0]])
Ptoy = softmax(Ytoy)
check("C-ex-p", np.allclose(Ptoy[0], [0.9, 0.1]) and np.allclose(Ptoy.mean(0), [0.7, 0.3]))
Pp2 = np.eye(2) - 0.5
yb = (Ytoy @ Pp2).mean(0)
check("C-ex-q", np.allclose(softmax(yb), [0.75, 0.25]))
exc = 0.5 * kl(Ptoy, softmax(yb)[None]).sum() - 0.5 * kl(Ptoy, Ptoy.mean(0)[None]).sum()
exact = 0.7 * np.log(14 / 15) + 0.3 * np.log(6 / 5)
check("C-ex-val", abs(exc - exact) < 1e-12 and exact > 0, f"{exc} {exact}")

print("checks:", COUNT)
print(f"part 3 excess: {exact:.6f}")
print("failures:", len(fails))
for f_ in fails[:20]:
    print("  ", f_)
