# serves: TV2
"""Exact checks of Part B.V (COARSE-THEORY.md): Fact F13, Lemma TV0, Theorem TV1, Corollary TV1.1, Theorem TV2,
Proposition TV3, Corollary TV3.1, Proposition TV4 part 3.

Instance: a planted pure-stream network (Definition DV1), stream width d = 14, two MLP blocks, C = 4 logits. In base
coordinates (then rotated by a random orthogonal Q, so no area is a set of coordinates):
  A_in   dims 0-3    written by the first write, read by both blocks
  A_mid  dims 4-6    written by block 1, read by block 2
  A_out  dims 7-12   written by blocks 1 and 2 and the first write, read only by the logits;
                     children dog {7, 8} and cat {9, 10}, own part {11, 12}
  loud   dim 13      written by the first write, read by no reader
Checks, per instance:
  A  Fact F13: LayerNorm and RMSNorm fold into n_eps and the readers; a rotation of the stream changes no logit
  B  Lemma TV0: the narrow network equals the projected full-width network; two bases give the same logits
  C  TV1 part 1 (no norm): dropping an unread subspace is exact
  D  TV1 part 2 (norm): unread loud line with the same energy everywhere -> exact with eps' = eps + c;
     varying energy -> not exact, and the reads differ exactly where the energy differs from c
  E  TV1 part 3: only the last energy varies -> logits = t (y_N - c_y), D_P <= both bounds
  F  TV1.1: varying content on one unread line, keeping the line -> exact
  G  TV2 (no norm): D_P(B^a) <= (sum sqrt eps_ro)^2 / 4 for random V and for TV3's V*; with a writer that leaks
     into a sibling area too; exact at k = C - 1 (TV3.1)
  H  TV2 (norm): planted dropped part with constant energy and eps_ro = 0 -> exact
  I  TV3: closed form = tail of singular values; random subspaces and a local search never beat it; nested in k
  J  TV4 part 3: children's cut error <= sum of children's sqrt errors, and >= TV3's optimum
  K  Lemma TV5: random read matrices with q_1 + q_2 >= d give K = {0}; the planted network has K = A_out + loud
  INFO  premise check: a block reader that reads a little of the cut area; the TV2 bound may then fail
Usage: python tv_regions.py [n_instances]
"""
import sys
import numpy as np

rng = np.random.default_rng(29)
N_INST = int(sys.argv[1]) if len(sys.argv) > 1 else 30
d, C, NX, DX = 14, 4, 300, 5
AREAS = {"in": [0, 1, 2, 3], "mid": [4, 5, 6], "out": list(range(7, 13)), "dog": [7, 8], "cat": [9, 10],
         "own": [11, 12], "loud": [13]}
fails = []


def check(name, ok, info=""):
    if not ok:
        fails.append(f"{name} {info}")


def logsm(y):
    m = y.max(-1, keepdims=True)
    return y - m - np.log(np.exp(y - m).sum(-1, keepdims=True))


def kl(p, yN, yB):
    a, b = logsm(yN), logsm(yB)
    return float((p * (np.exp(a) * (a - b)).sum(-1)).sum())


def nrm(h, eps, on=True):
    if not on:
        return h
    return h / np.sqrt((h * h).sum(-1, keepdims=True) + eps)


def basis(Q, idx):
    return Q[:, idx]


def proj(M):  # orthogonal projection onto the column space of M (columns orthonormal or not)
    if M.shape[1] == 0:
        return np.zeros((M.shape[0], M.shape[0]))
    u, sv, _ = np.linalg.svd(M, full_matrices=False)    # rank-aware: QR of dependent columns adds stray directions
    u = u[:, sv > 1e-9 * max(1.0, sv.max())]
    return u @ u.T


def make_net(Q, leak=0.0, readleak=0.0, loud_const=True, block_writes_out=True):
    """Planted network in rotated coordinates. Returns a dict of weights (DV1) and the first write e(x)."""
    B_ = lambda name: basis(Q, AREAS[name])
    q1, q2 = 8, 8
    E_in = rng.normal(size=(4, DX))
    X = rng.normal(size=(NX, DX))
    e = (X @ E_in.T) @ B_("in").T
    e += (np.tanh(X[:, :1]) * 0.7) @ rng.normal(size=(1, 6)) @ B_("out").T   # first write also writes into A_out
    loud = 6.0 * (np.sign(X[:, 1:2]) if loud_const else (1.5 + np.tanh(X[:, 1:2])))
    e += loud @ B_("loud").T
    R1 = rng.normal(size=(q1, 4)) @ B_("in").T
    R2 = rng.normal(size=(q2, 4)) @ B_("in").T + rng.normal(size=(q2, 3)) @ B_("mid").T
    if readleak:
        R2 = R2 + readleak * rng.normal(size=(q2, 6)) @ B_("out").T
    W1 = B_("mid") @ rng.normal(size=(3, q1))
    if block_writes_out:
        W1 = W1 + B_("out") @ rng.normal(size=(6, q1))
    W2 = (B_("out") @ rng.normal(size=(6, q2))) if block_writes_out else np.zeros((d, q2))
    if leak:  # a writer of block 2 that writes into dog also leaks into the sibling area cat
        W2[:, 0] = B_("dog") @ rng.normal(size=2) + leak * (B_("cat") @ rng.normal(size=2))
    U = rng.normal(size=(C, 6)) @ B_("out").T + 0.5 * rng.normal(size=(C, 4)) @ B_("in").T
    return dict(e=e, R=[R1, R2], b=[rng.normal(size=q1) * .3, rng.normal(size=q2) * .3], W=[W1, W2],
                c=[np.zeros(d), np.zeros(d)], U=U, cy=rng.normal(size=C), eps=[1.0, 1.0, 1.0])


def run(net, Pi=None, eps=None, norm=True, U=None, ret_reads=False):
    """Full-width forward pass; with Pi, every write is projected (Lemma TV0's ht)."""
    eps = net["eps"] if eps is None else eps
    Pi = np.eye(d) if Pi is None else Pi
    U = net["U"] if U is None else U
    h = net["e"] @ Pi.T
    hs, reads = [h], []
    for l in range(2):
        r = nrm(h, eps[l], norm) @ net["R"][l].T
        reads.append(r)
        g = np.tanh(r + net["b"][l])
        h = h + (g @ net["W"][l].T + net["c"][l]) @ Pi.T
        hs.append(h)
    y = nrm(h, eps[2], norm) @ U.T + net["cy"]
    return (y, hs, reads) if ret_reads else y


def run_narrow(net, Bm, eps, norm=True, U=None):
    """The projection cut N_D as a network of width d' (Definition DV2)."""
    U = net["U"] if U is None else U
    h = net["e"] @ Bm
    for l in range(2):
        g = np.tanh(nrm(h, eps[l], norm) @ (net["R"][l] @ Bm).T + net["b"][l])
        h = h + g @ (Bm.T @ net["W"][l]).T + Bm.T @ net["c"][l]
    return nrm(h, eps[2], norm) @ (U @ Bm).T + net["cy"]


def ortho_basis_of_complement(Dm):
    P = proj(Dm)
    w, v = np.linalg.eigh(np.eye(d) - P)
    return v[:, w > 0.5]


Pperp = np.eye(C) - np.ones((C, C)) / C


def readout_fit(nmat, S, V, U, p):
    """eps_ro(S, V) of DM12 and its least-norm minimiser L (C x d)."""
    target = nmat @ (U @ proj(S)).T @ Pperp.T            # rows: P_perp U P_S n(x)
    if V.shape[1] == 0:
        return float((p * (target ** 2).sum(-1)).sum()), np.zeros((C, d))
    X = nmat @ proj(V).T                                  # rows: P_V n(x)
    sw = np.sqrt(p)[:, None]
    Lt, *_ = np.linalg.lstsq(sw * X, sw * target, rcond=None)
    L = Pperp @ Lt.T
    res = target - X @ L.T @ Pperp.T
    return float((p * (res ** 2).sum(-1)).sum()), L


def tv3(nmat, S, U, p, k):
    """Proposition TV3: V*_k and the tail value."""
    QS, _ = np.linalg.qr(S)
    c = nmat @ QS
    Sig = (c * p[:, None]).T @ c
    w, v = np.linalg.eigh(Sig)
    w = np.clip(w, 0, None)
    Sh = (v * np.sqrt(w)) @ v.T
    Shp = (v * np.where(w > 1e-12, 1 / np.sqrt(np.where(w > 1e-12, w, 1)), 0)) @ v.T
    M = Pperp @ U @ QS
    _, sv, vt = np.linalg.svd(M @ Sh)
    keep = [i for i in range(min(k, len(sv))) if sv[i] > 1e-10]
    V = QS @ (Shp @ vt[keep].T) if keep else np.zeros((d, 0))
    return V, float((sv[k:] ** 2).sum()), sv


for inst in range(N_INST):
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    B_ = lambda name: basis(Q, AREAS[name])
    p = rng.dirichlet(np.ones(NX))
    net = make_net(Q)
    yN = run(net)

    # A. Fact F13: LayerNorm with gain and shift, folded; and a rotation.
    gam, bet, eps0 = rng.uniform(.5, 2, d), rng.normal(size=d), 1e-3
    Cm = np.eye(d) - np.ones((d, d)) / d

    def ln(h):
        hc = h @ Cm.T
        return gam * hc / np.sqrt((hc * hc).sum(-1, keepdims=True) / d + eps0) + bet

    def run_ln(e, W, c):
        h = e
        for l in range(2):
            g = np.tanh(ln(h) @ net["R"][l].T + net["b"][l])
            h = h + g @ W[l].T + c[l]
        return ln(h) @ net["U"].T + net["cy"]
    cvec = [rng.normal(size=d) * .2, rng.normal(size=d) * .2]
    y_ln = run_ln(net["e"], net["W"], cvec)
    Rf = [np.sqrt(d) * net["R"][l] * gam for l in range(2)]
    Uf = np.sqrt(d) * net["U"] * gam
    folded = dict(e=net["e"] @ Cm.T, R=Rf, b=[net["b"][l] + net["R"][l] @ bet for l in range(2)],
                  W=[Cm @ w for w in net["W"]], c=[Cm @ cc for cc in cvec], U=Uf, cy=net["cy"] + net["U"] @ bet,
                  eps=[d * eps0] * 3)
    check("A-layernorm-fold", np.abs(run(folded) - y_ln).max() < 1e-9, np.abs(run(folded) - y_ln).max())
    Qr, _ = np.linalg.qr(rng.normal(size=(d, d)))
    rot = dict(net, e=net["e"] @ Qr.T, R=[r @ Qr.T for r in net["R"]], W=[Qr @ w for w in net["W"]],
               c=[Qr @ cc for cc in net["c"]], U=net["U"] @ Qr.T)
    check("A-rotation", np.abs(run(rot) - yN).max() < 1e-10)

    # B. Lemma TV0: narrow = projected full width; two bases agree.
    Dm = rng.normal(size=(d, 3))
    Bm = ortho_basis_of_complement(Dm)
    Pi = np.eye(d) - proj(Dm)
    eps2 = [0.7, 1.3, 0.4]
    y1 = run_narrow(net, Bm, eps2)
    y2 = run_narrow(net, Bm @ np.linalg.qr(rng.normal(size=(d - 3, d - 3)))[0], eps2)
    y3 = run(net, Pi=Pi, eps=eps2)
    check("B-TV0", max(np.abs(y1 - y2).max(), np.abs(y1 - y3).max()) < 1e-10)

    # C. TV1 part 1: no norm, unread D = loud line plus part of A_out's complement that U does not read.
    D_unread = B_("loud")
    BmC = ortho_basis_of_complement(D_unread)
    check("C-TV1.1-nonorm", np.abs(run_narrow(net, BmC, net["eps"], norm=False) - run(net, norm=False)).max() < 1e-10)

    # D. TV1 part 2: norm. Loud line has energy 36 at every point (sign only).
    PD = proj(D_unread)
    yN_full, hs, readsN = run(net, ret_reads=True)
    cl = [float((p * ((hh @ PD) ** 2).sum(-1)).sum()) for hh in hs]
    eps_c = [net["eps"][l] + cl[l] for l in range(3)]
    yD = run_narrow(net, BmC, eps_c)
    check("D-TV1.2-exact", np.abs(yD - yN).max() < 1e-9, np.abs(yD - yN).max())
    net_v = make_net(Q, loud_const=False)
    yNv, hsv, readsNv = run(net_v, ret_reads=True)
    clv = [float((p * ((hh @ PD) ** 2).sum(-1)).sum()) for hh in hsv]
    yDv, _, readsDv = run(net_v, Pi=np.eye(d) - PD, eps=[net_v["eps"][l] + clv[l] for l in range(3)], ret_reads=True)
    check("D-TV1.2-varying-not-exact", kl(p, yNv, yDv) > 1e-6, kl(p, yNv, yDv))
    e1 = ((hsv[0] @ PD) ** 2).sum(-1)
    diff = np.abs(readsDv[0] - readsNv[0]).max(-1)
    same = np.abs(e1 - clv[0]) < 1e-9
    check("D-TV1.2-converse", (diff[~same] > 1e-9).all() and (diff[same] < 1e-9).all())

    # E. TV1 part 3: unread line written with varying size by the last block only.
    net_e = make_net(Q)
    net_e["e"] = net_e["e"] - net_e["e"] @ PD               # nothing in D before block 2 writes
    net_e["W"][1] = net_e["W"][1] + B_("loud") @ (2.0 * rng.normal(size=(1, 8)))
    yNe, hse, _ = run(net_e, ret_reads=True)
    eL = ((hse[2] @ PD) ** 2).sum(-1)
    cL = float((p * eL).sum())
    epsE = [net_e["eps"][0], net_e["eps"][1], net_e["eps"][2] + cL]
    yDe = run(net_e, Pi=np.eye(d) - PD, eps=epsE)
    hL, PiE = hse[2], np.eye(d) - PD
    t = np.sqrt(((hL ** 2).sum(-1) + net_e["eps"][2]) / (((hL @ PiE) ** 2).sum(-1) + net_e["eps"][2] + cL))
    check("E-temperature", np.abs((yDe - net_e["cy"]) - t[:, None] * (yNe - net_e["cy"])).max() < 1e-9)
    nL = nrm(hL, net_e["eps"][2])
    Aprime = ((hL @ PiE) ** 2).sum(-1) + net_e["eps"][2] + cL
    b1 = 0.25 * float((p * (eL - cL) ** 2 * ((nL @ (Pperp @ net_e["U"]).T) ** 2).sum(-1) / Aprime ** 2).sum())
    b2 = np.linalg.norm(Pperp @ net_e["U"], 2) ** 2 * float((p * (eL - cL) ** 2).sum()) / (4 * Aprime.min() ** 2)
    dpe = kl(p, yNe, yDe)
    check("E-bounds", dpe <= b1 + 1e-12 and b1 <= b2 + 1e-12, (dpe, b1, b2))

    # F. TV1.1: D = the 'own' area (2 dims), read by no reader; its content varies but lies on one line l0.
    net_f = make_net(Q, block_writes_out=False)
    net_f["U"] = net_f["U"] - net_f["U"] @ proj(B_("own"))           # the logits do not read 'own'
    l0 = B_("own") @ np.linalg.qr(rng.normal(size=(2, 1)))[0]
    net_f["e"] = net_f["e"] - net_f["e"] @ proj(B_("own")) + (1.5 + np.tanh(rng.normal(size=(NX, 1)))) @ l0.T
    yNf = run(net_f)
    D2 = B_("own")
    Dminus = D2 - proj(l0) @ D2
    yKeep = run(net_f, Pi=np.eye(d) - proj(Dminus), eps=net_f["eps"])
    yDrop = run(net_f, Pi=np.eye(d) - proj(D2),
                eps=[net_f["eps"][l] + float((p * ((hh @ proj(D2)) ** 2).sum(-1)).sum())
                     for l, hh in enumerate(run(net_f, ret_reads=True)[1])])
    check("F-TV1.1-keep-line-exact", np.abs(yKeep - yNf).max() < 1e-10, np.abs(yKeep - yNf).max())
    check("F-TV1.1-drop-all-not-exact", kl(p, yNf, yDrop) > 1e-8, kl(p, yNf, yDrop))

    # G. TV2 in the no-norm form, with and without a leaking writer.
    for leak in (0.0, 0.8):
        netg = make_net(Q, leak=leak)
        yg, hsg, _ = run(netg, norm=False, ret_reads=True)
        nmat = hsg[2]
        groups = {"dog": B_("dog"), "cat": B_("cat"), "own": B_("own")}
        for trial in range(3):
            Vs, Ls, sq = {}, {}, 0.0
            for name, S in groups.items():
                k = int(rng.integers(0, 2))
                V = S @ np.linalg.qr(rng.normal(size=(2, 2)))[0][:, :k] if trial else tv3(nmat, S, netg["U"], p, k)[0]
                er, L = readout_fit(nmat, S, V, netg["U"], p)
                Vs[name], Ls[name], sq = V, L, sq + np.sqrt(er)
            Dm = np.concatenate([S - proj(Vs[n]) @ S for n, S in groups.items()], 1)
            Pi = np.eye(d) - proj(Dm)
            PR = np.eye(d) - proj(B_("out"))
            Unew = netg["U"] @ PR + sum(Ls[n] @ proj(Vs[n]) for n in groups)
            yB = run(netg, Pi=Pi, norm=False, U=Unew)
            check("G-TV2-bound", kl(p, yg, yB) <= sq ** 2 / 4 + 1e-12, (leak, kl(p, yg, yB), sq ** 2 / 4))
    # TV3.1: area 'out' (6 dims) cut to C - 1 = 3 directions is exact in the no-norm form.
    netg = make_net(Q)
    yg, hsg, _ = run(netg, norm=False, ret_reads=True)
    V3, val, _ = tv3(hsg[2], B_("out"), netg["U"], p, C - 1)
    er, L = readout_fit(hsg[2], B_("out"), V3, netg["U"], p)
    Dm = B_("out") - proj(V3) @ B_("out")
    yB = run(netg, Pi=np.eye(d) - proj(Dm), norm=False,
             U=netg["U"] @ (np.eye(d) - proj(B_("out"))) + L @ proj(V3))
    check("G-TV3.1-exact", er < 1e-18 and kl(p, yg, yB) < 1e-12 and V3.shape[1] <= C - 1, (er, kl(p, yg, yB)))

    # H. TV2 with norms: dropped part = own area, written by the first write only with energy 4 at every point,
    #    logits read 'out' only through dog + cat (eps_ro = 0).
    neth = make_net(Q, block_writes_out=False)
    Xs = np.sign(rng.normal(size=(NX, 1)))
    neth["e"] = neth["e"] - neth["e"] @ proj(B_("out")) + (Xs @ np.array([[2.0, 0.0]])) @ B_("own").T \
        + rng.normal(size=(NX, 4)) @ basis(Q, [7, 8, 9, 10]).T
    neth["U"] = rng.normal(size=(C, 4)) @ basis(Q, [7, 8, 9, 10]).T + 0.5 * rng.normal(size=(C, 4)) @ B_("in").T
    yh, hsh, _ = run(neth, ret_reads=True)
    S, V = B_("out"), basis(Q, [7, 8, 9, 10])
    nmat = nrm(hsh[2], neth["eps"][2])
    er, L = readout_fit(nmat, S, V, neth["U"], p)
    PDh = proj(B_("own"))
    ch = [float((p * ((hh @ PDh) ** 2).sum(-1)).sum()) for hh in hsh]
    var = max(float((p * (((hh @ PDh) ** 2).sum(-1) - c) ** 2).sum()) for hh, c in zip(hsh, ch))
    yB = run(neth, Pi=np.eye(d) - PDh, eps=[neth["eps"][l] + ch[l] for l in range(3)],
             U=neth["U"] @ (np.eye(d) - proj(S)) + L @ proj(V))
    check("H-TV2-norm-exact", er < 1e-18 and var < 1e-18 and kl(p, yh, yB) < 1e-12, (er, var, kl(p, yh, yB)))

    # I. TV3: closed form vs random subspaces and a local search; nesting.
    netI = make_net(Q)
    _, hsI, _ = run(netI, ret_reads=True)
    nI = nrm(hsI[2], netI["eps"][2])
    S = B_("out")
    prevV = None
    for k in range(0, 4):
        Vk, val, sv = tv3(nI, S, netI["U"], p, k)
        er, _ = readout_fit(nI, S, Vk, netI["U"], p)
        check("I-TV3-value", abs(er - val) < 1e-9 * max(1, val), (k, er, val))
        for _ in range(20):
            Vr = S @ np.linalg.qr(rng.normal(size=(6, 6)))[0][:, :k]
            check("I-TV3-random", readout_fit(nI, S, Vr, netI["U"], p)[0] >= val - 1e-9)
        if k:
            Z = rng.normal(size=(6, k))
            best = readout_fit(nI, S, S @ np.linalg.qr(Z)[0], netI["U"], p)[0]
            for _ in range(300):
                Zn = Z + 0.2 * rng.normal(size=Z.shape)
                v = readout_fit(nI, S, S @ np.linalg.qr(Zn)[0], netI["U"], p)[0]
                if v < best:
                    Z, best = Zn, v
            check("I-TV3-search", best >= val - 1e-9, (k, best, val))
        if prevV is not None and prevV.shape[1]:
            check("I-TV3-nested", np.abs(proj(Vk) @ prevV - prevV).max() < 1e-8)
        prevV = Vk

    # J. TV4 part 3: children's cut.
    kids = {"dog": B_("dog"), "cat": B_("cat"), "own": B_("own")}
    for _ in range(5):
        parts, sq = [], 0.0
        for n, Sk in kids.items():
            kk = int(rng.integers(0, 3))
            Vk = Sk @ np.linalg.qr(rng.normal(size=(2, 2)))[0][:, :kk]
            parts.append(Vk)
            sq += np.sqrt(readout_fit(nI, Sk, Vk, netI["U"], p)[0])
        V = np.concatenate(parts, 1)
        er, _ = readout_fit(nI, S, V, netI["U"], p)
        opt = tv3(nI, S, netI["U"], p, V.shape[1])[1]
        check("J-TV4.3", np.sqrt(er) <= sq + 1e-12 and er >= opt - 1e-9, (er, sq ** 2, opt))

for _ in range(200):  # K. Lemma TV5
    qa, qb = int(rng.integers(1, d)), 0
    qb = int(rng.integers(max(0, d - qa), d))
    Rbar = np.concatenate([rng.normal(size=(qa, d)), rng.normal(size=(qb, d))], 0)
    if qa + qb >= d:
        check("K-TV5-generic", np.linalg.matrix_rank(Rbar) == d)
Qk, _ = np.linalg.qr(rng.normal(size=(d, d)))
netk = make_net(Qk)
Rbar = np.concatenate(netk["R"], 0)
_, svk, vtk = np.linalg.svd(Rbar)
Kb = vtk[np.linalg.matrix_rank(Rbar):].T
target = np.concatenate([basis(Qk, AREAS["out"]), basis(Qk, AREAS["loud"])], 1)
check("K-TV5-planted", Kb.shape[1] == 7 and np.abs(proj(Kb) - proj(target)).max() < 1e-9)

info = []
for inst in range(10):  # INFO: a block reader that reads a little of the cut area (eps_br > 0)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    p = rng.dirichlet(np.ones(NX))
    netr = make_net(Q, readleak=1.0)
    yr, hsr, _ = run(netr, norm=False, ret_reads=True)
    S = basis(Q, AREAS["out"])
    V, val, _ = tv3(hsr[2], S, netr["U"], p, C - 1)
    er, L = readout_fit(hsr[2], S, V, netr["U"], p)
    Dm = S - proj(V) @ S
    yB = run(netr, Pi=np.eye(d) - proj(Dm), norm=False, U=netr["U"] @ (np.eye(d) - proj(S)) + L @ proj(V))
    info.append((kl(p, yr, yB), er / 4))

print(f"instances {N_INST}; failures {len(fails)}")
for f in fails[:20]:
    print("  FAIL", f)
print("INFO read leak (premise of TV2 broken): D_P vs TV2 bound, 10 instances:")
print("  " + ", ".join(f"{a:.3g} vs {b:.3g}" for a, b in info))
print(f"  bound fails in {sum(a > b + 1e-12 for a, b in info)} of 10")
