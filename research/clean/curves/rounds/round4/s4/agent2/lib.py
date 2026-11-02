# serves: DQ19
import numpy as np, torch
from scipy.cluster.hierarchy import linkage
torch.set_num_threads(2)
R='rounds/round4/s4/agent2/'
sb=np.load(R+'sb.npy'); A=np.load(R+'A.npy')
C=1000
def nest(Z,n=1000):
    nodes=list(range(n))
    for x,y,_,_ in Z: nodes.append([nodes[int(x)],nodes[int(y)]])
    return nodes[-1]
def lca_of(tree):
    lca=-np.ones((C,C),dtype=np.int64); cnt=[0]
    def rec(n):
        if isinstance(n,int): return [n]
        ps=[rec(x) for x in n]; t=cnt[0]; cnt[0]+=1
        for i in range(len(ps)):
            for j in range(i+1,len(ps)):
                lca[np.ix_(ps[i],ps[j])]=t; lca[np.ix_(ps[j],ps[i])]=t
        return [x for p in ps for x in p]
    rec(tree); np.fill_diagonal(lca,0); return torch.from_numpy(lca),cnt[0]
def pseudo(tree,fitrows,testrows):
    """same fit as the tool, but fit on fitrows (subset of A) and test on testrows (subset of A)"""
    torch.set_default_dtype(torch.float64)
    s=torch.from_numpy(sb).clamp_min(1e-12); s=s/s.sum(1,keepdim=True); L=s.log()
    lca,nn=lca_of(tree); off=~torch.eye(C,dtype=torch.bool); w=s*off
    F=torch.from_numpy(np.asarray(fitrows)); T=torch.from_numpy(np.asarray(testrows))
    beta=torch.zeros(C); fv=torch.zeros(C,C); wA,LA=w[F],L[F]; idxA=lca[F].reshape(-1); idx=lca.reshape(-1)
    for _ in range(40):
        alpha=(wA*(LA-beta[None]-fv[F])).sum(1)/wA.sum(1)
        beta=(wA*(LA-alpha[:,None]-fv[F])).sum(0)/wA.sum(0).clamp_min(1e-300)
        r=LA-alpha[:,None]-beta[None]
        num=torch.zeros(nn).index_add_(0,idxA,(wA*r).reshape(-1)); den=torch.zeros(nn).index_add_(0,idxA,wA.reshape(-1))
        f=num/den.clamp_min(1e-300); fv=f[idx].reshape(C,C)
    out={}
    for nm,rows in (('fit',F),('test',T)):
        wR,LR=w[rows],L[rows]
        al=(wR*(LR-beta[None]-fv[rows])).sum(1)/wR.sum(1)
        r=LR-al[:,None]-beta[None]-fv[rows]
        out[nm]=float(((wR*r**2).sum()/wR.sum()).sqrt())
    return out
def splits(seed=1):
    p=np.random.default_rng(seed).permutation(A); return p[:250],p[250:]
