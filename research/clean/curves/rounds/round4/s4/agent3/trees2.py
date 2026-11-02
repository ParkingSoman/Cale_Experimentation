# serves: DQ19
import sys, subprocess, json
sys.path.insert(0, '/Users/shonusengupta/Cale_Experimentation/research/clean/curves/rounds/round4/s4/agent3')
from common import *
from scipy.cluster.hierarchy import linkage
sb, meta, lab = load()
A = splitA()
K, C = sb.shape
Wc = meta["W"].astype(np.float64)
D = sys.path[0]


def nested(Z):
    nodes = list(range(C))
    for x, y, _, _ in Z:
        nodes.append([nodes[int(x)], nodes[int(y)]])
    return nodes[-1]


def score(name, tree):
    p = f"{D}/t_{name}.json"
    json.dump(tree, open(p, "w"))
    out = subprocess.run(["/Users/shonusengupta/Cale_Experimentation/research/clean/curves/.venv/bin/python",
                          "/Users/shonusengupta/Cale_Experimentation/research/clean/curves/pc_r4_s4_eval.py",
                          "--net", "resnet18", "--tree", p, "--split", "A"], capture_output=True, text=True).stdout
    print(name, json.loads(out)["rms"], flush=True)


# column profile of each class on A rows (rows scaled away): log s[A, i] - alpha_c, centered over A
off = ~torch.eye(K, C, dtype=torch.bool)
W = (sb * off)[A].numpy(); L = sb.log()[A].numpy()
al = np.zeros(len(A)); be = np.zeros(C)
for _ in range(40):
    al = (W * (L - be[None])).sum(1) / W.sum(1)
    be = (W * (L - al[:, None])).sum(0) / np.maximum(W.sum(0), 1e-300)
R = (L - al[:, None] - be[None])
Wr = np.sqrt(W) * R
u, s, vt = np.linalg.svd(Wr, full_matrices=False)
Wn = Wc / np.linalg.norm(Wc, axis=1, keepdims=True)
Wcen = Wc - Wc.mean(0)
cands = {
    "avg_cos_cen": linkage(Wcen, "average", metric="cosine"),
    "wgt_cos": linkage(Wn, "weighted", metric="cosine"),
    "avg_euc": linkage(Wc, "average"),
    "cent_unit": linkage(Wn, "centroid"),
    "comp_cos": linkage(Wn, "complete", metric="cosine"),
}
for name, Z in cands.items():
    score(name, nested(Z))
