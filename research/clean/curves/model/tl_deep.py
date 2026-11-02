# serves: TL3
# serves: TL4
# serves: TL5
"""Theorems TL3, TL5 and Proposition TL4 (COARSE-THEORY.md B.L.2) on synthetic deep networks built so that the
claims of TL3 hold at eps = 0 (3 hidden layers, a tree of areas, gates, units sheltered at 0 or at a negative value).
A  every claim of TL3's premises has eps 0, and eps_lin = 0 at every node (TL3); negative control: one unit that
   switches inside its area gives eps_lin > 0 at its ancestors.
B  TL4: Sigma_u = sum of the children's Sigma + Gamma_u (with the silent part), E likewise, at every node with children.
C  TL5 along a deep zoom (leaf groups -> merge -> lift -> lift with merge): the value from the table rules equals the
   value of the always-on box of TP1 part 2 evaluated on N's data, at every level and size.
D  no box of any shape found by a local optimiser (random restarts) goes below the closed form (TP1 part 1).
A failure is a mismatch above 1e-8 (or an optimiser value below the closed form by more than 1e-7)."""
import json
import os

import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(1029)
fails = []
out = {}
NY = 5
PP = np.eye(NY) - np.ones((NY, NY)) / NY

# tree: node -> (layer, leaves, parent)
TREE = {
    "A": (1, ["l1", "l2", "l3", "l5"], None), "B": (1, ["l4"], None),
    "A1": (2, ["l1", "l2"], "A"), "A2": (2, ["l3"], "A"), "B1": (2, ["l4"], "B"),
    "A11": (3, ["l1"], "A1"), "A12": (3, ["l2"], "A1"), "A21": (3, ["l3"], "A2"), "B11": (3, ["l4"], "B1"),
}
LEAVES = ["l1", "l2", "l3", "l4", "l5"]          # l5 lies in A only: silent below layer 1
D_IN = 3


def children(v):
    return [c for c, (_, _, p) in TREE.items() if p == v]


def build(switch_bug=False):
    pts = []
    for li, lf in enumerate(LEAVES):
        for _ in range(30):
            pts.append((li, rng.uniform(0.5, 3.0, D_IN)))
    for _ in range(4):
        pts.append((-1, np.zeros(D_IN)))         # points in no area of layer 1 (silent at the root)
    n = len(pts)
    p = rng.uniform(0.5, 1.5, n)
    p /= p.sum()
    C = np.zeros((n, len(LEAVES)))
    F = {lf: np.zeros((n, D_IN)) for lf in LEAVES}
    for i, (li, f) in enumerate(pts):
        if li >= 0:
            C[i, li] = 1
            F[LEAVES[li]][i] = f
    # units: list of dicts with node, kind ('gate', 'carrier', 'feat'), home leaves, values
    units = {1: [], 2: [], 3: []}
    vals = {}

    def ind(leaves, carriers):
        return sum(carriers[lf] for lf in leaves)

    for layer in (1, 2, 3):
        for v, (lv, leaves, par) in TREE.items():
            if lv != layer:
                continue
            kids = children(v)
            # one feature block per leaf, and one per child with several leaves (it varies only on that home)
            homes = [[lf] for lf in leaves] + [TREE[c][1] for c in kids if len(TREE[c][1]) > 1]
            if layer == 1:
                carriers = {lf: C[:, LEAVES.index(lf)] for lf in LEAVES}
                featsrc = {(lf,): F[lf] for lf in LEAVES}                  # inputs, 0 off their leaf
            else:
                carriers = {u["leaf"]: vals[u["id"]] for u in units[layer - 1] if u["node"] == par and u["kind"] == "carrier"}
                featsrc = {}
                for u in units[layer - 1]:
                    if u["node"] == par and u["kind"] == "feat":
                        featsrc.setdefault(tuple(u["home"]), []).append(vals[u["id"]])
                featsrc = {k: np.stack(vv, 1) for k, vv in featsrc.items()}
            # gate
            pre = ind(leaves, carriers)
            uid = f"{v}.g"
            units[layer].append(dict(id=uid, node=v, kind="gate", pre=pre, home=leaves))
            vals[uid] = np.maximum(pre, 0)
            # carriers
            for lf in leaves:
                uid = f"{v}.c.{lf}"
                units[layer].append(dict(id=uid, node=v, kind="carrier", leaf=lf, pre=carriers[lf].copy(), home=[lf]))
                vals[uid] = np.maximum(carriers[lf], 0)
            # features, one block per home (a child's leaves), sheltered at 0 or at -gamma
            for hi, H in enumerate(homes):
                # read only sources that vary inside the home H (so the unit is 0 off H and sheltered)
                src = np.concatenate([featsrc[k] for k in featsrc if set(k) <= set(H)], axis=1)
                for j in range(3):
                    m = rng.normal(0, 1, src.shape[1])
                    beta = 1 + np.abs(src @ m).max()
                    gamma = 0.0 if j == 0 else rng.uniform(0.5, 2)
                    pre = src @ m + beta * ind(H, carriers) + gamma * ind(leaves, carriers) - gamma
                    if switch_bug and v == "A1" and hi == 0 and j == 1:
                        on = ind(H, carriers) > 0
                        med = np.median((src @ m)[on])
                        pre = src @ m - med * ind(H, carriers)              # switches at half its home's points
                    uid = f"{v}.f{hi}.{j}"
                    units[layer].append(dict(id=uid, node=v, kind="feat", pre=pre, home=list(H)))
                    vals[uid] = np.maximum(pre, 0)
    top = units[3]
    H3 = np.stack([vals[u["id"]] for u in top], 1)
    Wout = rng.normal(0, 1, (NY, H3.shape[1]))
    c0 = rng.normal(0, 1, NY)
    Y = (H3 @ Wout.T + c0) @ PP
    area = {v: (C[:, [LEAVES.index(lf) for lf in leaves]].sum(1) > 0) for v, (_, leaves, _) in TREE.items()}
    X = np.concatenate([C] + [F[lf] for lf in LEAVES], axis=1)
    return dict(n=n, p=p, Y=Y, units=units, vals=vals, area=area, X=X, Wout=Wout, c0=c0)


def node_units(net, v):
    lv = TREE[v][0]
    return [u for u in net["units"][lv] if u["node"] == v]


def claims(net):
    """errors of the district, shelter, on-inside and judgment claims (all nodes)."""
    p, worst = net["p"], 0.0
    for v in TREE:
        R = net["area"][v]
        for u in node_units(net, v):
            z, h = u["pre"], np.maximum(u["pre"], 0)
            if (p * h).sum() > 0:
                worst = max(worst, (p[~R] * h[~R]).sum() / (p * h).sum())                    # district
            if np.ptp(z[~R]) > 0:
                worst = max(worst, float(np.ptp(z[~R])))                                    # shelter (any spread)
            if u["kind"] == "gate":
                worst = max(worst, float(np.ptp(h[R])), float(np.abs(1 - h[R]).max()))      # flat, judgment
            elif (p[R] * z[R] ** 2).sum() > 0:
                worst = max(worst, (p[R] * np.minimum(z[R], 0) ** 2).sum() / (p[R] * z[R] ** 2).sum())
    return worst


def summary(net, v):
    """(Sigma, E, eps_lin, P, ybar, B, top ids) of node v; free terms = constants on R_v (the gate is flat)."""
    p, Y, R = net["p"], net["Y"], net["area"][v]
    T = [u for u in node_units(net, v) if u["kind"] != "gate"]
    Z = np.stack([u["pre"] for u in T], 1)
    w = p[R] / p[R].sum()
    Yt, Zt = Y[R] - w @ Y[R], Z[R] - w @ Z[R]
    Czz = (Zt * p[R][:, None]).T @ Zt
    Cyz = (Yt * p[R][:, None]).T @ Zt
    B = Cyz @ np.linalg.pinv(Czz)
    r = Yt - Zt @ B.T
    E = float((p[R][:, None] * Yt ** 2).sum())
    return dict(S=B @ Czz @ B.T, E=E, eps=float((p[R][:, None] * r ** 2).sum()), P=float(p[R].sum()),
                ybar=w @ Y[R], B=B, T=T, Z=Z)


def topk(S, k):
    ev = np.sort(np.linalg.eigvalsh(S))[::-1]
    return ev[:k].clip(min=0).sum()


# ---------- A: claims and TL3 ----------
net = build()
out["A_claims_worst"] = claims(net)
if out["A_claims_worst"] > 1e-12:
    fails.append(f"A: a claim of TL3 has error {out['A_claims_worst']}")
summ = {v: summary(net, v) for v in TREE}
out["A_eps_lin"] = {v: s["eps"] / max(s["E"], 1e-30) for v, s in summ.items()}
for v, e in out["A_eps_lin"].items():
    if e > 1e-10:
        fails.append(f"A: eps_lin({v}) relative {e}")
bad = build(switch_bug=True)
sb = {v: summary(bad, v)["eps"] / summary(bad, v)["E"] for v in ("A", "A1")}
out["A_control_switch_inside"] = dict(claims_worst=claims(bad), eps_lin_rel=sb)

# ---------- B: TL4 ----------
b_err = 0.0
for u_ in TREE:
    kids = children(u_)
    if not kids:
        continue
    su = summ[u_]
    R = net["area"][u_]
    Rk = np.zeros(net["n"], bool)
    S_sum, E_sum, Gam = np.zeros((NY, NY)), 0.0, np.zeros((NY, NY))
    for c in kids:
        S_sum += summ[c]["S"]
        E_sum += summ[c]["E"]
        d = summ[c]["ybar"] - su["ybar"]
        Gam += summ[c]["P"] * np.outer(d, d)
        Rk |= net["area"][c]
    R0 = R & ~Rk
    if R0.any():
        y0 = net["Y"][R0]
        if np.ptp(y0, axis=0).max() > 1e-10:
            fails.append(f"B: y_N not constant on the silent part of {u_}")
        d = y0[0] - su["ybar"]
        Gam += net["p"][R0].sum() * np.outer(d, d)
    b_err = max(b_err, float(np.abs(su["S"] - S_sum - Gam).max()) / su["E"], abs(su["E"] - E_sum - np.trace(Gam)) / su["E"])
out["B_TL4_max_relative_error"] = b_err
if b_err > 1e-8:
    fails.append(f"B: TL4 error {b_err}")


# ---------- C: TL5 along a deep zoom ----------
def table(v):
    """Sigma_v and E_v from the tables of the deepest nodes by the rules of TL4 (no regression above layer 3)."""
    kids = children(v)
    if not kids:
        return summ[v]["S"], summ[v]["E"]
    S, E = np.zeros((NY, NY)), 0.0
    R, Rk = net["area"][v], np.zeros(net["n"], bool)
    ybar = (net["p"][R][:, None] * net["Y"][R]).sum(0) / net["p"][R].sum()
    for c in kids:
        Sc, Ec = table(c)
        S += Sc
        E += Ec
        Pc = net["p"][net["area"][c]].sum()
        yc = (net["p"][net["area"][c]][:, None] * net["Y"][net["area"][c]]).sum(0) / Pc
        S += Pc * np.outer(yc - ybar, yc - ybar)
        E += Pc * float((yc - ybar) @ (yc - ybar))
        Rk |= net["area"][c]
    R0 = R & ~Rk
    if R0.any():
        P0 = net["p"][R0].sum()
        d = net["Y"][R0][0] - ybar
        S += P0 * np.outer(d, d)
        E += P0 * float(d @ d)
    return S, E


def box_value(nodes, k):
    """terms of a group with nodes W_S, for the always-on box of TP1 part 2 evaluated on N's data."""
    S = sum(summ[w]["S"] for w in nodes)
    ev, V = np.linalg.eigh(S)
    Q = V[:, np.argsort(-ev)][:, :k]
    tot = 0.0
    for w in nodes:
        R, sw = net["area"][w], summ[w]
        pre = sw["Z"] @ sw["B"].T @ Q                      # read B_w^T q on T_w (other nodes constant on R_w)
        C = pre + 1 + np.abs(pre).max(0)
        res = (net["Y"][R] - C[R] @ Q.T) @ PP
        res = res - (net["p"][R] / net["p"][R].sum()) @ res
        tot += 0.5 * float((net["p"][R][:, None] * res ** 2).sum())
    return tot


zoom = {"G1 leaf groups (parent A1)": [["A11"], ["A12"]], "G2 merge (parent A1)": [["A11", "A12"]],
        "G3 lift (parent A)": [["A1"]], "G4 lift (parent root)": [["A"]]}
c_rows = {}
for name, groups in zoom.items():
    for k in (1, 2, 3, 4):
        val_table = sum(0.5 * (sum(table(w)[1] for w in g) - topk(sum(table(w)[0] for w in g), k)) for g in groups)
        val_box = sum(box_value(g, k) for g in groups)
        c_rows[f"{name}, k={k}"] = dict(table=val_table, box=val_box)
        if abs(val_table - val_box) > 1e-8:
            fails.append(f"C: {name} k={k}: table {val_table} != box {val_box}")
out["C_TL5"] = c_rows
vals_k = {k: c_rows[f"G4 lift (parent root), k={k}"]["table"] for k in (1, 2, 3, 4)}
if any(vals_k[k + 1] > vals_k[k] + 1e-12 for k in (1, 2, 3)):
    fails.append("C: least not monotone in k")


# ---------- D: no box beats the closed form ----------
def general_box_min(v, k, restarts=12):
    R, sw = net["area"][v], summ[v]
    Z, Y, p = sw["Z"][R], net["Y"][R], net["p"][R]
    w = p / p.sum()
    m = Z.shape[1]

    def f(th):
        M = th[: k * m].reshape(k, m)
        b = th[k * m: k * m + k]
        V = th[k * m + k:].reshape(NY, k)
        res = (np.maximum(Z @ M.T + b, 0) @ V.T - Y) @ PP
        res = res - w @ res
        return 0.5 * float((p[:, None] * res ** 2).sum())

    best = np.inf
    for _ in range(restarts):
        th0 = rng.normal(0, 0.5, k * m + k + NY * k)
        r = minimize(f, th0, method="L-BFGS-B", options=dict(maxiter=3000))
        best = min(best, r.fun)
    return best


d_rows = {}
for v in ("A1", "A"):
    for k in (1, 2):
        lin = 0.5 * (summ[v]["E"] - topk(summ[v]["S"], k))
        g = general_box_min(v, k)
        d_rows[f"{v}, k={k}"] = dict(closed_form=lin, optimiser_best=g)
        if g < lin - 1e-7:
            fails.append(f"D: optimiser {g} below closed form {lin} at {v} k={k}")
out["D_no_box_beats"] = d_rows

# ---------- E: Theorem TL6, the bound with errors ----------
def layer_mats(net):
    """N's weight matrices, recovered exactly from the pre-activations (they are affine in the layer below)."""
    prev, mats = net["X"], []
    for layer in (1, 2, 3):
        Z = np.stack([u["pre"] for u in net["units"][layer]], 1)
        A = np.concatenate([prev, np.ones((net["n"], 1))], 1)
        sol = np.linalg.lstsq(A, Z, rcond=None)[0]
        if np.abs(A @ sol - Z).max() > 1e-8:
            fails.append(f"E: layer {layer} not affine in the layer below")
        mats.append((sol[:-1].T, sol[-1]))
        prev = np.maximum(Z, 0)
    return mats


def forward(net, mats):
    prev, Zs, Hs = net["X"], [], []
    for W, b in mats:
        Z = prev @ W.T + b
        Zs.append(Z)
        Hs.append(np.maximum(Z, 0))
        prev = Hs[-1]
    Y = (Hs[-1] @ net["Wout"].T + net["c0"]) @ PP
    return Zs, Hs, Y


def eps_lin_general(net, v, Zs, Hs, Y):
    """Definition DP1 with F_w = span(1, h_J) on R_w (J = the node's gate)."""
    lv, p, R = TREE[v][0], net["p"], net["area"][v]
    idx = [i for i, u in enumerate(net["units"][lv]) if u["node"] == v]
    gi = [i for i in idx if net["units"][lv][i]["kind"] == "gate"]
    ti = [i for i in idx if net["units"][lv][i]["kind"] != "gate"]
    F = np.concatenate([np.ones((R.sum(), 1)), Hs[lv - 1][R][:, gi]], 1)
    sw = np.sqrt(p[R])[:, None]
    Q, _ = np.linalg.qr(F * sw)

    def proj(M):
        Mw = M * sw
        return Mw - Q @ (Q.T @ Mw)

    Yt, Zt = proj(Y[R]), proj(Zs[lv - 1][R][:, ti])
    B = np.linalg.lstsq(Zt, Yt, rcond=None)[0]
    return float(((Yt - Zt @ B) ** 2).sum())


def tl6_bound(net, w, Zs, Hs):
    lw, p, Rw = TREE[w][0], net["p"], net["area"][w]

    def nrm(f, mask):
        m = mask & Rw
        if not m.any():
            return 0.0
        return float(np.sqrt((p[m] * np.asarray(f)[m] ** 2).sum()))

    below = {v for v in TREE if v == w or (TREE[v][0] > lw and set(TREE[v][1]) <= set(TREE[w][1])
                                            and anc(v, w))}
    eps = None
    iota = {w: 0.0}
    for layer in range(lw, 4):
        ds, cterm = [], 0.0
        for i, u in enumerate(net["units"][layer]):
            z, h = Zs[layer - 1][:, i], Hs[layer - 1][:, i]
            v = u["node"]
            if v not in below:
                ds.append(nrm(h, np.ones(net["n"], bool)))
                continue
            Rv = net["area"][v]
            off = Rw & ~Rv
            if u["kind"] == "gate" and v == w:
                ds.append(0.0)                                   # J_w: the stand-in keeps h (Theorem TL6, item 2)
            elif u["kind"] == "gate":
                kap = (p[Rv] * h[Rv]).sum() / p[Rv].sum()
                ds.append(nrm(h - kap, Rv) + nrm(h, off))
            else:
                zet = (p[off] * z[off]).sum() / p[off].sum() if off.any() else 0.0
                ds.append(nrm(np.minimum(z, 0), Rv) + nrm(h, off) + nrm(z - zet, off))
        Delta = float(np.linalg.norm(ds))
        if layer == lw:
            eps = Delta
            continue
        Wn = np.linalg.norm(mats_p[layer - 1][0], 2)
        for v in below:
            if TREE[v][0] != layer:
                continue
            par = TREE[v][2]
            Rv = net["area"][v]
            uidx = [i for i, u in enumerate(net["units"][layer]) if u["node"] == v]
            off = Rw & ~Rv
            cvec = []
            for i in uidx:
                z, h = Zs[layer - 1][:, i], Hs[layer - 1][:, i]
                if net["units"][layer][i]["kind"] == "gate":
                    cvec.append((p[Rv] * h[Rv]).sum() / p[Rv].sum())
                else:
                    cvec.append(abs((p[off] * z[off]).sum() / p[off].sum()) if off.any() else 0.0)
            if Rv.sum() == net["area"][par].sum():
                iota[v] = iota[par]
            else:
                k = [i for i in uidx if net["units"][layer][i]["kind"] == "gate"][0]
                z, h = Zs[layer - 1][:, k], Hs[layer - 1][:, k]
                kap = (p[Rv] * h[Rv]).sum() / p[Rv].sum()
                zet = (p[off] * z[off]).sum() / p[off].sum()
                if kap <= 0 or kap <= zet:                       # no unit k_v as TL6 requires
                    return np.inf
                g = nrm(kap - h, Rv) + nrm(np.minimum(z, 0), Rv) + nrm(z - zet, off)
                iota[v] = (g + Wn * eps) / (kap - zet)
            cterm += float(np.linalg.norm(cvec)) * iota[v]
        eps = Delta + Wn * eps + cterm
    return np.linalg.norm(PP @ net["Wout"], 2) ** 2 * eps ** 2


def anc(v, w):
    while v is not None:
        if v == w:
            return True
        v = TREE[v][2]
    return False


mats0 = layer_mats(net)
e_rows = {}
for sigma in (0.0, 1e-4, 1e-3, 1e-2, 5e-2):
    mats_p = [(W + sigma * rng.normal(0, 1, W.shape) * (np.abs(W).max()), b + sigma * rng.normal(0, 1, b.shape))
              for W, b in mats0]
    Zs, Hs, Yp = forward(net, mats_p)
    for w in ("A", "A1", "B"):
        el = eps_lin_general(net, w, Zs, Hs, Yp)
        bd = tl6_bound(net, w, Zs, Hs)
        e_rows[f"sigma={sigma}, w={w}"] = dict(eps_lin=el, bound=bd)
        if el > bd * (1 + 1e-9) + 1e-12:
            fails.append(f"E: sigma={sigma} w={w}: eps_lin {el} above the TL6 bound {bd}")
out["E_TL6"] = e_rows

out["failures"] = fails
print(json.dumps(out, indent=1, default=float))
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "logs"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "..", "logs", "tl_deep.json"), "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print("FAILURES:", len(fails))
