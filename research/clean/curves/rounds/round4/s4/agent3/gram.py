# serves: DQ19
"""Claim form: log sbar_c[i] = alpha_c + t_c * beta_i + sum_g theta_g m_cg m_ig   (off diagonal), m = one vector per class.
Shared numbers (beta, m, theta) fitted on rows A1 (half of A). Held-out: rows A2 (other half of A): m_c for c in A2 is the
class's COLUMN vector learned from A1 rows. alpha_c, t_c refitted per held-out row (closed form). Weights sbar."""
import sys, json, argparse
sys.path.insert(0, '/Users/shonusengupta/Cale_Experimentation/research/clean/curves/rounds/round4/s4/agent3')
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--temp", type=int, default=1)      # row temperature t_c on/off
ap.add_argument("--nonneg", type=int, default=0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--l2", type=float, default=1e-4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--net", default="resnet18")
ap.add_argument("--feat", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
sb, meta, lab = load(a.net)
K, C = sb.shape
A = splitA(K)
rng = np.random.default_rng(7)
pa = rng.permutation(A)
A1, A2 = np.sort(pa[:len(A) // 2]), np.sort(pa[len(A) // 2:])
L = sb.log()
off = ~torch.eye(K, C, dtype=torch.bool)
Wt = sb * off


def fit_rows(Lr, Wr, F, beta, temp):
    """closed-form alpha_c (and t_c) per row, given F=shared part"""
    T = Lr - F
    if not temp:
        al = (Wr * (T - beta[None])).sum(1) / Wr.sum(1)
        return al, torch.ones(len(Lr), dtype=torch.double)
    s0 = Wr.sum(1); s1 = (Wr * beta[None]).sum(1); s2 = (Wr * beta[None] ** 2).sum(1)
    y0 = (Wr * T).sum(1); y1 = (Wr * T * beta[None]).sum(1)
    det = s0 * s2 - s1 ** 2
    t = (s0 * y1 - s1 * y0) / det
    al = (y0 - t * s1) / s0
    return al, t


def shared(E, th, beta, rows, t):
    M = E[rows]
    G = (M * th[None]) @ E.T
    return G


def rms(rows, al, t, E, th, beta):
    G = shared(E, th, beta, rows, t)
    pred = al[:, None] + t[:, None] * beta[None] + G
    r = L[rows] - pred
    w = Wt[rows]
    return float(((w * r ** 2).sum() / w.sum()).sqrt())


# init from SVD of alpha+beta residual on A1
W1 = Wt[A1]; L1 = L[A1]
al = torch.zeros(len(A1)).double(); be = torch.zeros(C).double()
for _ in range(40):
    al = (W1 * (L1 - be[None])).sum(1) / W1.sum(1)
    be = (W1 * (L1 - al[:, None])).sum(0) / W1.sum(0).clamp_min(1e-300)
R = L1 - al[:, None] - be[None]
Wn = torch.from_numpy(meta["W"].astype(np.float64)); Wn = (Wn - Wn.mean(0)) / Wn.std()
if a.feat:
    P = (torch.randn(Wn.shape[1], a.k, dtype=torch.double) * 0.02).requires_grad_()
    E = P
else:
    E = (torch.randn(C, a.k, dtype=torch.double) * 0.1).requires_grad_()
def emb():
    X = (Wn @ E) if a.feat else E
    return torch.nn.functional.softplus(X) if a.nonneg else X
th = torch.ones(a.k, dtype=torch.double).requires_grad_()
beta = be.clone().requires_grad_()
alp = al.clone().requires_grad_()
tp = torch.ones(len(A1), dtype=torch.double).requires_grad_()
opt = torch.optim.Adam([E, th, beta, alp, tp], lr=0.03)
A1t = torch.from_numpy(A1)
for step in range(a.steps):
    opt.zero_grad()
    Ee = emb()
    thh = th.abs() if a.nonneg else th
    G = (Ee[A1t] * thh[None]) @ Ee.T
    tt = tp if a.temp else torch.ones_like(tp)
    pred = alp[:, None] + tt[:, None] * beta[None] + G
    r = L1 - pred
    loss = (W1 * r ** 2).sum() / W1.sum() + a.l2 * (E ** 2).mean()
    loss.backward(); opt.step()
    if step % 300 == 0 or step == a.steps - 1:
        print(step, "train rms", float(((W1 * r.detach() ** 2).sum() / W1.sum()).sqrt()), flush=True)
with torch.no_grad():
    Ee = emb()
    thh = th.abs() if a.nonneg else th
    A2t = torch.from_numpy(A2)
    G2 = (Ee[A2t] * thh[None]) @ Ee.T
    al2, t2 = fit_rows(L[A2t], Wt[A2t], G2, beta.detach(), a.temp)
    out = dict(k=a.k, temp=a.temp, nonneg=a.nonneg, train=float(((W1 * r.detach() ** 2).sum() / W1.sum()).sqrt()),
               heldout_A2=rms(A2t, al2, t2, Ee, thh, beta.detach()))
    # control: same without Gram term (alpha + t*beta only)
    al0, t0 = fit_rows(L[A2t], Wt[A2t], torch.zeros(len(A2), C, dtype=torch.double), beta.detach(), a.temp)
    pred0 = al0[:, None] + t0[:, None] * beta.detach()[None]
    r0 = L[A2t] - pred0; w0 = Wt[A2t]
    out["no_gram_A2"] = float(((w0 * r0 ** 2).sum() / w0.sum()).sqrt())
    out["mass_within3"] = float((Wt[A2t] * ((L[A2t] - (al2[:, None] + t2[:, None] * beta.detach()[None] + G2)).abs() <= np.log(3))).sum() / Wt[A2t].sum())
print(json.dumps(out))
np.savez(sys.path[0] + f"/gram_k{a.k}_t{a.temp}_n{a.nonneg}_f{a.feat}.npz", E=Ee.detach().numpy(), th=thh.detach().numpy(), beta=beta.detach().numpy())
