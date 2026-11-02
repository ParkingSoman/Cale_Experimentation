# serves: DQ19
# local copy of the A-side fit of pc_r4_s4_eval.py, returns residuals (A rows only)
import numpy as np, torch
R='/Users/shonusengupta/Cale_Experimentation/research/clean/curves/'
torch.set_default_dtype(torch.float64)
Y=np.load(R+'cache/map/resnet18_y.npy'); meta=np.load(R+'cache/map/resnet18_meta.npz')
lab=torch.from_numpy(meta['labels'].astype(np.int64))
S=torch.softmax(torch.from_numpy(Y.astype(np.float64)),1)
sb=torch.stack([S[lab==c].mean(0) for c in range(1000)]).clamp_min(1e-12); sb=sb/sb.sum(1,keepdim=True)
K=C=1000; L=sb.log()
perm=np.random.default_rng(2026).permutation(K); A=torch.from_numpy(perm[:500])
def lca_of(tree):
    lca=-np.ones((C,C),dtype=np.int64); cnt=[0]; size=[]
    def rec(n):
        if isinstance(n,int): return [n]
        parts=[rec(x) for x in n]; t=cnt[0]; cnt[0]+=1
        for i in range(len(parts)):
            for j in range(i+1,len(parts)):
                lca[np.ix_(parts[i],parts[j])]=t; lca[np.ix_(parts[j],parts[i])]=t
        out=[x for p in parts for x in p]; size.append(len(out)); return out
    rec(tree); np.fill_diagonal(lca,0); return torch.from_numpy(lca),cnt[0],np.array(size)
def fitA(tree):
    lca,nn,size=lca_of(tree)
    off=~torch.eye(K,C,dtype=torch.bool); w=sb*off
    beta=torch.zeros(C); fv=torch.zeros(K,C); wA,LA=w[A],L[A]; idxA=lca[A].reshape(-1)
    for _ in range(60):
        alpha=(wA*(LA-beta[None]-fv[A])).sum(1)/wA.sum(1)
        beta=(wA*(LA-alpha[:,None]-fv[A])).sum(0)/wA.sum(0).clamp_min(1e-300)
        r=LA-alpha[:,None]-beta[None]
        num=torch.zeros(nn).index_add_(0,idxA,(wA*r).reshape(-1)); den=torch.zeros(nn).index_add_(0,idxA,wA.reshape(-1))
        f=num/den.clamp_min(1e-300); fv=f[lca.reshape(-1)].reshape(K,C)
    alpha=(wA*(LA-beta[None]-fv[A])).sum(1)/wA.sum(1)
    r=LA[:, :]-alpha[:,None]-beta[None]-fv[A]
    return r,wA,lca[A],size
