"""Closure of n101's system (LOG "Closure"). Mac, numpy only. Bar and definitions are in the LOG, written before this ran.

Sets: S0 (gate s3:69 + 6 core children), S1 (+ gate s3:78 + its Floor A children), S2 (S1 + ring of 11 + n212),
S3 (the |S2| A neurons that co-fire most with n101 on the 19,000 held + test photos).
M1 weight match (sum of reads), M2 top-photo union on the 16,000 train-cache photos, M3 peak correlation there,
M4 share of n101's positive read on gates + members' condition channels. Controls: 200 random same-size sets.
Gate data for s3:78 comes from the cluster run oc_gate.py 78 (results/oc_gate78.npz); without it S1 = S0 + gate only.
Saved: results/oc_closure.npz
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
RN = os.path.join(HERE, "..", "..", "neuron", "results", "coarse", "r18")
rng = np.random.default_rng(0)

W = np.load(os.path.join(R, "oc_weights.npz"))
act = np.load(os.path.join(RN, "read_s3.npz"))["pos"].astype(np.float32).mean(0)          # typical activity, 256
RA = W["read_all_A"].sum((2, 3)) * act                                                     # 512 x 256
RC = W["read_c101"][0].sum((1, 2)) * act                                                   # 256
FEED = np.load(os.path.join(RN, "path_A.npz"))["feed0"]                                   # 512 x 256
FL = np.load(os.path.join(R, "oc_floor.npz"))
TP, CP = FL["tpos"].astype(np.float32), FL["cpos"].astype(np.float32)                    # train-cache peaks
S1D = np.load(os.path.join(R, "oc_step1.npz"))
APOS, C101 = S1D["A_pos"].astype(np.float32), S1D["B1_pos"][:, 101].astype(np.float32)    # held + test peaks

CORE = [23, 97, 163, 275, 297, 305]
RING = [252, 108, 37, 237, 385, 71, 123, 103, 81, 415, 298]
TOPN = 190


def floor_a(gate, ppk, c, pre):
    """Floor A child rule for a gate channel (oc_floorA.py, all photos, top 3)."""
    rank = (FEED > FEED[:, [gate]]).sum(1)
    top = np.argsort(-ppk)[:200]
    drive = (np.clip(c[top], 0, None) * (pre[top] > 0)).mean(0)
    return sorted(np.where((rank < 3) & (drive > np.quantile(drive, .90)))[0].tolist())


def cond_channels(members, gates):
    out = set()
    for j in members:
        f = FEED[j].copy(); f[list(gates)] = -np.inf
        out |= set(np.argsort(-f)[:3].tolist())
    return out


ZT = (TP - TP.mean(0)) / (TP.std(0) + 1e-6)
TOPS = np.argsort(-TP, 0)[:TOPN]                                                           # 190 x 512
CTOP = set(np.argsort(-CP)[:TOPN].tolist())


def measures(members, gates):
    m = list(members)
    s = RA[m].sum(0)
    m1 = np.corrcoef(RC, s)[0, 1]
    m1y = np.corrcoef(RC, np.clip(RA[m], 0, None).sum(0))[0, 1]
    union = set(TOPS[:, m].ravel().tolist())
    m2 = len(CTOP & union) / TOPN
    m3 = np.corrcoef(CP, ZT[:, m].mean(1))[0, 1]
    pos = np.clip(RC, 0, None)
    ch = set(gates) | cond_channels(m, gates)
    m4 = pos[list(ch)].sum() / pos.sum()
    return np.array([m1, m2, m3, m4, m1y]), union


def random_ctrl(size, gates, n=200):
    return np.array([measures(rng.choice(512, size, replace=False), gates)[0] for _ in range(n)])


# check the child rule reproduces Floor A for s3:69
kids69 = floor_a(69, S1D["ppk"].astype(np.float32), S1D["A_c69"].astype(np.float32), S1D["A_pre"].astype(np.float32))
print("Floor A children of s3:69 (check, expect the 6 core):", kids69)

g78 = os.path.join(R, "oc_gate78.npz")
if not os.path.exists(g78):
    g78 = os.path.expanduser("~/uva-gpu/inbox/oc_gate78.npz")
if os.path.exists(g78):
    G = np.load(g78)
    assert (G["ix"] == S1D["ix"]).all()
    kids78 = floor_a(78, G["ppk"].astype(np.float32), G["A_c"].astype(np.float32), G["A_pre"].astype(np.float32))
else:
    kids78 = []
    print("NOTE: oc_gate78.npz missing; S1 has no s3:78 children")
print("Floor A children of s3:78:", kids78)

S0 = CORE
S1 = sorted(set(CORE) | set(kids78))
S2 = sorted(set(S1) | set(RING) | {212})
cof = np.array([np.corrcoef(C101, APOS[:, j])[0, 1] for j in range(512)])
S3 = sorted(np.argsort(-cof)[:len(S2)].tolist())
sets = {"S0": (S0, [69]), "S1": (S1, [69, 78]), "S2": (S2, [69, 78]), "S3": (S3, [69, 78])}

print(f"\n{'set':4} {'size':>4} | {'M1':>5} {'M2':>5} {'M3':>5} {'M4':>5} | rand99 M1 M2 M3 | sum-yes M1 | closes")
res = {}
for name, (mem, gates) in sets.items():
    v, _ = measures(mem, gates)
    rc = random_ctrl(len(mem), gates)
    q = np.quantile(rc, .99, 0)
    c1 = v[0] >= .7 and v[1] >= .7
    c2 = (v[:3] > q[:3]).all()
    res[name] = dict(members=np.array(mem), v=v, q99=q)
    print(f"{name:4} {len(mem):4d} | {v[0]:5.2f} {v[1]:5.2f} {v[2]:5.2f} {v[3]:5.2f} | {q[0]:5.2f} {q[1]:5.2f} {q[2]:5.2f} | {v[4]:5.2f} | C1 {c1} C2 {c2}")

print("\nS3 members (co-firing):", S3)
print("S3 inside S2:", len(set(S3) & set(S2)), "of", len(S3))
print("top co-firing A neurons:", [(int(j), round(float(cof[j]), 2)) for j in np.argsort(-cof)[:10]])

# C5 material: n101's positive read left outside the best set's gates + condition channels
best = max(res, key=lambda k: res[k]["v"][0] + res[k]["v"][1])
mem, gates = sets[best]
ch = set(gates) | cond_channels(mem, gates)
pos = np.clip(RC, 0, None)
out = [c for c in np.argsort(-pos) if c not in ch][:10]
print(f"\nbest set {best}; n101's largest positive read outside it (channel, share):",
      [(int(c), round(float(pos[c] / pos.sum()), 3)) for c in out])
print("n101's top positive channels overall:", [(int(c), round(float(pos[c] / pos.sum()), 3)) for c in np.argsort(-pos)[:10]])

np.savez(os.path.join(R, "oc_closure.npz"), kids69=kids69, kids78=np.array(kids78), cof=cof,
         **{f"{k}_members": r["members"] for k, r in res.items()},
         **{f"{k}_v": r["v"] for k, r in res.items()}, **{f"{k}_q99": r["q99"] for k, r in res.items()})
