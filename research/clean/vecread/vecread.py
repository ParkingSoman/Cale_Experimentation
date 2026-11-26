"""Read a vector in ResNet-18 (A) against a dictionary of Broden concept directions, the way word vectors are read
against known words. See README.md.

Spaces: s3 (stage-3 output, per cell, 256), s3k (stage-3 3 x 3 patches as stage 4 reads them, 2304),
s4b0 (stage-4 stream after block 0, before ReLU, 512), s4b1 (final stream before ReLU, 512).

Two kinds of vector:
  write-like (a shift of the activations: a write, a fold's write, a concept direction) - read as it is.
  read-like  (a measurement: a read row, a read kernel, a probe) - first turned into its pattern, cov @ r
             (Haufe et al. 2014: the activation change a linear read responds to), then read like a write.
Reading = score against every concept direction (both poles), a 5-term signed decomposition with its leftover share,
and the concepts over-represented on the held-out cells where the vector's projection is highest (IoU).
Metric (tests K4, K8, confirmed on untouched photos): writes = raw cosine; reads = whitened pattern, z-score against
random reads (hub correction).

    dic = Dictionary("s4b0"); held = Held()
    read_vector(dic, v, kind="write", held=held)
    read_vector(Dictionary("s3k"), r.reshape(-1), kind="read", held=held)     # r: 256 x 3 x 3 read kernel
    compare(dic, a, b, kind, held)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SPACES = {"s3": (256, 14), "s3k": (2304, 7), "s4b0": (512, 7), "s4b1": (512, 7)}
LGRID = {"s3": 14, "s3k": 14, "s4b0": 7, "s4b1": 7}
MAPS = {"s3": "s3", "s3k": "s3", "s4b0": "s4b0", "s4b1": "s4b1"}
CATS = ("color", "object", "part", "material", "scene", "texture")
PHOTO_LEVEL = ("scene", "texture")
MIN_CELLS, MIN_PHOTOS, MIN_PHOTOS_PL = 50, 10, 20
TERMS = 5          # decomposition terms, fixed before any run (plan, 2026-11-23)
TO7 = -np.ones(196, int)
TO7[[(2 * i) * 14 + 2 * j for i in range(7) for j in range(7)]] = np.arange(49)


def _labels():
    import broden
    return broden.load_labels()


def _within_source(S, N, T, M):
    """S: (ns, C) sums on the entry's cells per source, N: (ns,), T, M: per-source totals. Count-weighted mean of
    (mean on the entry's cells - mean on the other cells of the same source)."""
    d, w = 0.0, 0.0
    for s in np.nonzero(N)[0]:
        d = d + N[s] * (S[s] / N[s] - (T[s] - S[s]) / max(M[s] - N[s], 1))
        w += N[s]
    return d / w


class Dictionary:
    """Concept directions for one space, from the build photos.
    Entry = one Broden label, or a merged group of label twins. Rule per category: 'half' unless that category keeps
    fewer than 10 labels on this grid, then 'centre'. Support: >= 50 cells and >= 10 photos (scene, texture: 20).
    Twins (decided on the two build halves only): labels c, k merge when cos(d_c, d_k) >= the lower of their own
    split-half cosines, i.e. k is as close to c as c's two halves are to each other. kept = passes K1 (vr_check)."""

    def __init__(self, space, tag=""):
        self.space = space
        self.C, self.G = SPACES[space]
        b = np.load(os.path.join(RES, f"dict_build{tag}.npz"))
        lab = _labels()
        ns = len(b["sources"])
        self.ns = ns
        n = b[f"{space}_n"]
        self.mu = b[f"{space}_X"] / n
        self.cov = b[f"{space}_XtX"] / n - np.outer(self.mu, self.mu)
        w, U = np.linalg.eigh(self.cov)
        w = np.maximum(w, w.max() * 1e-6)
        self.W = (U / np.sqrt(w)) @ U.T            # cov^-1/2, for the whitened metric
        lam = 0.1 * w.mean()                       # shrinkage fixed before K8 (2026-11-23)
        self.Ws = (U / np.sqrt(w + lam)) @ U.T     # (cov + lam I)^-1/2, the shrunk metric
        T = [b[f"{space}_T_h{h}"] for h in (0, 1)]
        M = [b[f"{space}_M_h{h}"] for h in (0, 1)]
        self.T, self.M = T[0] + T[1], M[0] + M[1]
        cand = []                                  # (label, rule, S_h0, S_h1, N_h0, N_h1)
        per_rule = {}
        for r in ("half", "centre"):
            S = [b[f"{space}_S_{r}_h{h}"].reshape(-1, ns, self.C) for h in (0, 1)]
            N = [b[f"{space}_N_{r}_h{h}"].reshape(-1, ns) for h in (0, 1)]
            P = b[f"{space}_P_{r}_h0"] + b[f"{space}_P_{r}_h1"]
            per_rule[r] = (S, N, P)
        self.rule_of_cat = {}
        for c in CATS:
            counts = {}
            for r in ("half", "centre"):
                S, N, P = per_rule[r]
                ok = [l for l in range(len(P)) if l in lab and lab[l][1] == c and self._support(N, P, l, c)]
                counts[r] = ok
            r = "half" if len(counts["half"]) >= 10 else "centre"
            self.rule_of_cat[c] = r
            S, N, P = per_rule[r]
            for l in counts[r]:
                cand.append((l, r, S[0][l], S[1][l], N[0][l], N[1][l]))
        # Tool metric = whitened (chosen by K3, run of 2026-11-23). Twins and reliability are scored in it, on the two
        # build halves only (held photos never used here).
        K = len(cand)
        u = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
        wm = lambda A: A @ self.W.T
        half = lambda S, N, h: _within_source(S, N, T[h], M[h]) if N.sum() else np.zeros(self.C)
        h0 = u(wm(np.array([half(c[2], c[4], 0) for c in cand])))
        h1 = u(wm(np.array([half(c[3], c[5], 1) for c in cand])))
        parent = list(range(K))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        # twins = mutual confusion across the halves: c's best match in the other half is k, and k's is c.
        # One-way confusion = an unreliable label (dropped below), not a twin.
        X01 = h0 @ h1.T
        best01, best10 = np.argmax(X01, axis=1), np.argmax(X01, axis=0)
        conf = [set() for _ in range(K)]
        for i in range(K):
            for j in (best01[i], best10[i]):
                if j != i:
                    conf[i].add(int(j))
        for i in range(K):
            for j in conf[i]:
                if i in conf[j]:
                    parent[find(i)] = find(j)
        groups = {}
        for i in range(K):
            groups.setdefault(find(i), []).append(i)
        groups = list(groups.values())
        # reliability: the entry's half-0 direction picks itself out among all half-1 directions, and the reverse
        G0 = u(wm(np.array([half(sum(cand[i][2] for i in g), sum(cand[i][4] for i in g), 0) for g in groups])))
        G1 = u(wm(np.array([half(sum(cand[i][3] for i in g), sum(cand[i][5] for i in g), 1) for g in groups])))
        Y = G0 @ G1.T
        reliable = (np.argmax(Y, axis=1) == np.arange(len(groups))) & (np.argmax(Y, axis=0) == np.arange(len(groups)))
        self.split_half = np.diag(Y)
        self.groups, D, names, cats, ncell, rules = [], [], [], [], [], []
        self.dropped = []
        for gi, g in enumerate(groups):
            g = sorted(g, key=lambda i: -(cand[i][4] + cand[i][5]).sum())
            nm = [lab[cand[i][0]][0] for i in g]
            name = " / ".join(nm[:3]) + (f" +{len(nm) - 3}" if len(nm) > 3 else "")
            if not reliable[gi]:
                self.dropped.append(name)
                continue
            S = sum(cand[i][2] + cand[i][3] for i in g)
            N = sum(cand[i][4] + cand[i][5] for i in g)
            D.append(_within_source(S, N, self.T, self.M))
            names.append(name)
            cats.append(lab[cand[g[0]][0]][1])
            ncell.append(N.sum())
            rules.append(cand[g[0]][1])
            self.groups.append(np.array([cand[i][0] for i in g]))
        self.split_half = self.split_half[reliable]
        self.D, self.names, self.cat = np.array(D), np.array(names), np.array(cats)
        self.n, self.rule = np.array(ncell), np.array(rules)
        self.key = np.array([int(g.min()) for g in self.groups])
        self.kept = np.ones(len(self.D), bool)
        kp = os.path.join(RES, f"kept_{space}{tag}.json")
        if os.path.exists(kp):
            ok = set(json.load(open(kp))["kept"])
            self.kept = np.array([int(k) in ok for k in self.key])

    @staticmethod
    def _support(N, P, l, c):
        cells = N[0][l].sum() + N[1][l].sum()
        need = MIN_PHOTOS_PL if c in PHOTO_LEVEL else MIN_PHOTOS
        return cells >= MIN_CELLS and P[l] >= need

    def metric(self, V, whiten):
        """whiten: False / "raw", True / "whitened", or "shrunk"."""
        V = np.atleast_2d(V)
        if whiten == "shrunk":
            return V @ self.Ws.T
        if whiten is True or whiten == "whitened":
            return V @ self.W.T
        return V

    def unit(self, whiten, only_kept=True):
        D = self.metric(self.D, whiten)
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
        return D[self.kept] if only_kept else D

    def read_null(self, whiten=True, n=2000):
        """Per-entry mean and sd of the cosine with the patterns of n random read vectors (seeded). Read hubs
        (sky, wall, painted: entries on the activations' largest directions) win for any read; a z-score against
        this null removes that (same idea as CSLS hub correction for word vectors, Conneau et al. 2018)."""
        key = ("null", whiten)
        if key not in self.__dict__.setdefault("_cache", {}):
            g = np.random.default_rng(12345).standard_normal((n, self.C)) @ self.cov
            U = self.metric(g, whiten)
            U /= np.linalg.norm(U, axis=1, keepdims=True)
            Cs = U @ self.unit(whiten).T
            self._cache[key] = (Cs.mean(0), Cs.std(0))
        return self._cache[key]

    def pattern(self, r):
        """Forward pattern of a read-like vector: the activation change it responds to (cov @ r)."""
        return self.cov @ np.asarray(r, float).reshape(-1)


# ------------------------------------------------------------------ geometry
def omp(v, D, k=TERMS):
    """Greedy signed fit of v by k unit atoms (orthogonal matching pursuit). Returns picks, coefs, leftover share."""
    r, picks, coef = v.copy(), [], np.zeros(0)
    for _ in range(k):
        j = int(np.argmax(np.abs(D @ r)))
        if j in picks:
            break
        picks.append(j)
        A = D[picks].T
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        r = v - A @ coef
    return picks, coef, float(r @ r / (v @ v))


def geometry(dic, v, whiten=False, top=8, kind="write"):
    """Write-like: rank by cosine. Read-like (v = pattern): rank by the per-entry z-score against random reads."""
    D = dic.unit(whiten)
    u = dic.metric(v, whiten)[0]
    u = u / np.linalg.norm(u)
    cos = D @ u
    score = cos
    if kind == "read":
        m, sd = dic.read_null(whiten)
        score = (cos - m) / sd
    names, cats = dic.names[dic.kept], dic.cat[dic.kept]
    o = np.argsort(-score)
    picks, coef, left = omp(u, D)
    return {"metric": whiten if isinstance(whiten, str) else ("whitened" if whiten else "raw"), "score": "z vs random reads" if kind == "read" else "cosine",
            "up": [(names[j], cats[j], round(float(score[j]), 3)) for j in o[:top]],
            "down": [(names[j], cats[j], round(float(score[j]), 3)) for j in o[::-1][:top]],
            "decomposition": [(names[j], cats[j], round(float(c), 3)) for j, c in zip(picks, coef)],
            "leftover": round(left, 3), "_top": int(np.nonzero(dic.kept)[0][o[0]])}


METRIC = {"write": "raw", "read": "whitened"}   # chosen by K8 (writes) and K4 (reads), confirmed on held set 2


def read_vector(dic, v, kind="write", held=None, whiten=None, top=8):
    whiten = METRIC[kind] if whiten is None else whiten
    v = np.asarray(v, float).reshape(-1)
    g = dic.pattern(v) if kind == "read" else v
    out = {"kind": kind}
    out.update(geometry(dic, g, whiten, top, kind))
    if held is not None:
        out["examples"] = examples(held, dic, held.project(dic.space, v))
    return out


def compare(dic, a, b, kind="write", held=None, whiten=None):
    a, b = np.asarray(a, float).reshape(-1), np.asarray(b, float).reshape(-1)
    ga, gb = (dic.pattern(a), dic.pattern(b)) if kind == "read" else (a, b)
    au, bu = a / np.linalg.norm(a), b / np.linalg.norm(b)
    out = {"cosine_raw": round(float(au @ bu), 3)}
    if kind == "read":
        out["cosine_of_patterns"] = round(float(ga @ gb / np.linalg.norm(ga) / np.linalg.norm(gb)), 3)
    else:
        out["cosine_whitened"] = round(float(_wcos(dic, a, b)), 3)   # reported only; the write metric is raw
    out["a"] = read_vector(dic, a, kind, held, whiten)
    out["b"] = read_vector(dic, b, kind, held, whiten)
    out["a_minus_b"] = read_vector(dic, au - bu, kind, held, whiten)
    return out


def _wcos(dic, a, b):
    x, y = dic.metric(a, True)[0], dic.metric(b, True)[0]
    return x @ y / np.linalg.norm(x) / np.linalg.norm(y)


# ------------------------------------------------------------------ held-out data
class Held:
    """Held-out Broden photos (val split): whole maps (fp16 memmap) + cell labels."""

    def __init__(self, tag=""):
        self.lab = np.load(os.path.join(RES, f"held_labels{tag}.npz"))
        self.n_img = len(self.lab["rows"])
        self.src = self.lab["source"]
        self.maps = {s: np.load(os.path.join(RES, f"held_{s}{tag}.npy"), mmap_mode="r") for s in ("s3", "s4b0", "s4b1")}
        self._cells = {}

    def label_cells(self, G, rule):
        """label -> (img, cell) unique pairs on grid G under the rule."""
        key = (G, rule)
        if key not in self._cells:
            L = self.lab
            keep = L[f"g{G}_cov"] >= 0.5 if rule == "half" else L[f"g{G}_centre"]
            img, cell, lab = L[f"g{G}_img"][keep], L[f"g{G}_cell"][keep], L[f"g{G}_label"][keep]
            big = self.n_img * G * G
            flat = np.unique(lab.astype(np.int64) * big + img.astype(np.int64) * G * G + cell)
            lab, rest = flat // big, flat % big
            d = {}
            bounds = np.nonzero(np.diff(lab))[0] + 1
            for seg in np.split(np.arange(len(lab)), bounds):
                if len(seg):
                    d[int(lab[seg[0]])] = (rest[seg] // (G * G), rest[seg] % (G * G))
            self._cells[key] = d
        return self._cells[key]

    def cells(self, dic, k):
        """(img, cell) pairs of dictionary entry k on the space's output grid (s3k: centre cells mapped to 7 x 7)."""
        LG = LGRID[dic.space]
        lc = self.label_cells(LG, dic.rule[k])
        parts = [lc[int(l)] for l in dic.groups[k] if int(l) in lc]
        if not parts:
            return np.zeros(0, int), np.zeros(0, int)
        img = np.concatenate([p[0] for p in parts])
        cell = np.concatenate([p[1] for p in parts])
        if dic.space == "s3k":
            cell = TO7[cell]
            ok = cell >= 0
            img, cell = img[ok], cell[ok]
        G = SPACES[dic.space][1]
        flat = np.unique(img * G * G + cell)
        return flat // (G * G), flat % (G * G)

    def project(self, space, V, imgs=None, chunk=500):
        """Raw projection of held cells onto one vector (C,) or several (k, C) -> (n, cells) or (n, cells, k).
        s3k vectors are 3 x 3 kernels on s3 maps (stride 2, padding 1)."""
        import torch
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        V = np.asarray(V, np.float32)
        single = V.ndim == 1
        V = np.atleast_2d(V)
        X = self.maps[MAPS[space]]
        imgs = np.arange(self.n_img) if imgs is None else np.asarray(imgs)
        Vt = torch.from_numpy(V).to(dev)
        out = []
        for a in range(0, len(imgs), chunk):
            x = torch.from_numpy(np.asarray(X[imgs[a:a + chunk]], np.float32)).to(dev)
            if space == "s3k":
                y = torch.nn.functional.conv2d(x, Vt.view(-1, 256, 3, 3), stride=2, padding=1)
            else:
                y = torch.einsum("nchw,kc->nkhw", x, Vt)
            out.append(y.flatten(2).permute(0, 2, 1).cpu().numpy())
        out = np.concatenate(out)
        return out[:, :, 0] if single else out


def examples(held, dic, values, imgs=None, q=0.99, top=5):
    """Entries over-represented on the top 1% held cells by projection (IoU, as in Network Dissection)."""
    imgs = np.arange(held.n_img) if imgs is None else np.asarray(imgs)
    pos = -np.ones(held.n_img, int)
    pos[imgs] = np.arange(len(imgs))
    hot = values >= np.quantile(values, q)
    n_hot = hot.sum()
    res = []
    for k in np.nonzero(dic.kept)[0]:
        im, ce = held.cells(dic, k)
        p = pos[im]
        ok = p >= 0
        if not ok.any():
            continue
        inter = hot[p[ok], ce[ok]].sum()
        res.append((inter / (n_hot + ok.sum() - inter), k))
    res.sort(reverse=True)
    return [(dic.names[k], dic.cat[k], round(float(i), 3)) for i, k in res[:top]]


def card_md(title, reading):
    lines = [f"# {title}", ""]
    for key, val in reading.items():
        if key.startswith("_"):
            continue
        if isinstance(val, list):
            lines += [f"**{key}**", "", "| concept | category | value |", "|---|---|---|"]
            lines += [f"| {n} | {c} | {x} |" for n, c, x in val] + [""]
        elif isinstance(val, dict):
            lines += [card_md(f"{title} / {key}", val).replace("# ", "## ", 1)]
        else:
            lines += [f"**{key}**: {val}", ""]
    return "\n".join(lines)


def gate_card(held, mask, imgs, top=8, min_cells=20, min_photos=10):
    """Read a SET OF CELLS (a gate map) from the data: Broden labels over-represented on it.
    mask: bool (len(imgs), 49), True on the cells of the set (7 x 7 grid, s4b0 cells). imgs: held photo indices.
    Cell-level labels (object, part, material, color; 'half' coverage rule): lift = cells of the set with the label /
    expected count if the set's cells were placed at random positions IN THE SAME PHOTOS (shuffle within photo).
    Photo-level labels (scene, texture) cover every cell of a photo, so a within-photo shuffle gives lift 1 by
    construction; they are scored per photo instead: share of the set's cells in photos with the label / share of
    all cells of imgs in those photos. Added 2026-11-25 (VECMATH 9, gate maps)."""
    L = held.lab
    names = _labels()
    imgs = np.asarray(imgs)
    pos = -np.ones(held.n_img, int)
    pos[imgs] = np.arange(len(imgs))
    mask = np.asarray(mask, bool)
    per_photo = mask.sum(1).astype(float)                    # set cells per photo
    n_set = per_photo.sum()
    keep = (L["g7_cov"] >= 0.5) & (pos[L["g7_img"]] >= 0)
    img, cell, lab = L["g7_img"][keep], L["g7_cell"][keep], L["g7_label"][keep]
    p = pos[img]
    flat = np.unique(lab.astype(np.int64) * (len(imgs) * 49) + p * 49 + cell)
    lab, rest = flat // (len(imgs) * 49), flat % (len(imgs) * 49)
    p, cell = rest // 49, rest % 49
    hit = mask[p, cell].astype(float)
    exp = per_photo[p] / 49.0
    out_cell, out_photo = [], []
    for l in np.unique(lab):
        s = lab == l
        nm, cat = names.get(int(l), ("?", "", {}))[:2]
        photos = np.unique(p[s])
        if len(photos) < min_photos:
            continue
        if cat in PHOTO_LEVEL:
            obs = per_photo[photos].sum()
            e = n_set * len(photos) / len(imgs)
            if obs >= min_cells and e > 0:
                out_photo.append((obs / e, nm, cat, int(obs)))
        else:
            obs, e = hit[s].sum(), exp[s].sum()
            if obs >= min_cells and e > 0:
                out_cell.append((obs / e, nm, cat, int(obs)))
    out_cell.sort(reverse=True)
    out_photo.sort(reverse=True)
    r = lambda v: [(nm, cat, round(float(li), 2), n) for li, nm, cat, n in v[:top]]
    return {"cells": int(n_set), "photos": int((per_photo > 0).sum()), "cell_level": r(out_cell),
            "photo_level": r(out_photo)}
