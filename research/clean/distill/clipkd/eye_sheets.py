"""Eye reading (plan step after a U1 fail): 10 KD student neurons, top-6 photos of the student neuron,
its best single teacher neuron, and its greedy 3-member teacher union (chosen on FIT, as union_check.py)."""
import os, sys
import numpy as np
from PIL import Image
R = '/Users/shonusengupta/Cale_Experimentation/research/clean/distill/clipkd'
sys.path.insert(0, R)
from score_clip import net
OUTD = sys.argv[1]
O = R + '/out/'
L = {n: np.load(O + f'maxfire_{n}.npz') for n in ('teacher', 'kd')}
pos = L['teacher']['pos']; ids = L['teacher']['photo_ids']; fit = pos < 4; q = 40
Tf = L['teacher']['fire'].astype(np.float32)
ph = net.Photos('read')
rng = np.random.default_rng(7)
def top(a, n): return np.argsort(-a)[:n]
def sheet(rows, path):
    W = 112
    im = Image.new('RGB', (6 * W, len(rows) * W), 'white')
    for r, idx in enumerate(rows):
        for c, i in enumerate(idx[:6]):
            im.paste(Image.fromarray(ph.raw(np.array([ids[i]]))[0]).resize((W, W)), (c * W, r * W))
    im.save(path)
for b in (7, 11):
    P = np.concatenate([Tf[:, b], Tf[:, b - 1]], 1)
    P = P / (np.percentile(P[fit], 99, axis=0) + 1e-6)
    S = L['kd']['fire'][:, b].astype(np.float32)
    for j in rng.choice(S.shape[1], 5, replace=False):
        sm = np.zeros(len(S), bool); sm[fit] = False
        sf = np.zeros(fit.sum(), bool); sf[top(S[fit, j], q)] = True
        Pf = P[fit]
        tm = np.zeros_like(Pf, bool); np.put_along_axis(tm, np.argpartition(-Pf, q - 1, 0)[:q], True, 0)
        ov = (tm & sf[:, None]).sum(0) + 1e-3 * Pf[sf].mean(0)
        cand = list(np.argsort(-ov)[:64]); mem = [cand[0]]; u = Pf[:, cand[0]].copy()
        for _ in range(2):
            best, bs = None, -1
            for c in cand:
                if c in mem: continue
                uu = np.maximum(u, Pf[:, c]); m = np.zeros(len(uu), bool); m[top(uu, q)] = True
                s = (m & sf).sum() + 1e-3 * uu[sf].mean()
                if s > bs: best, bs = c, s
            mem.append(best); u = np.maximum(u, Pf[:, best])
        rows = [top(S[:, j], 6), top(P[:, mem[0]], 6)] + [top(P[:, m], 6) for m in mem[1:]] + [top(P[:, mem].max(1), 6)]
        name = f'b{b+1}_kd{j}_t' + '-'.join(f'{"b" if m < 3072 else "b-1:"}{m % 3072}' for m in mem)
        sheet(rows, os.path.join(OUTD, name + '.png'))
        print(name, 'FIT overlap single', int((tm[:, mem[0]] & sf).sum()), 'of', q)
