# serves: DQ19
import numpy as np, torch, sys
from lib import *
m=np.load('cache/map/resnet18_meta.npz'); W=torch.from_numpy(m['W']).double()
Ws=W/ W.norm(dim=1,keepdim=True) if False else W
sp=[splits(1),splits(2)]
s=torch.from_numpy(sb).clamp_min(1e-12); s=s/s.sum(1,keepdim=True); L=s.log()
off=~torch.eye(C,dtype=torch.bool); w=s*off
def learn(rows,r,ridge=1e-3,steps=600,seed=0):
    torch.manual_seed(seed)
    F=torch.from_numpy(rows)
    U=(0.05*torch.randn(512,r)).double().requires_grad_()
    beta=torch.zeros(C).double().requires_grad_()
    alpha=torch.zeros(len(F)).double().requires_grad_()
    g0=torch.zeros(1).double().requires_grad_()
    opt=torch.optim.Adam([U,beta,alpha],lr=0.02)
    wF,LF=w[F],L[F]; wsum=wF.sum()
    for it in range(steps):
        E=W@U                       # 1000 x r
        P=E[F]@E.T                   # symmetric affinity
        r_=LF-alpha[:,None]-beta[None]-P
        loss=(wF*r_**2).sum()/wsum+ridge*(U**2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return (W@U).detach().numpy(), loss.item()
if __name__=='__main__':
    for r in (8,24,64):
        res=[]
        for (a,b_) in sp:
            E,l=learn(a,r)
            res.append(pseudo(nest(linkage(E,'ward')),a,b_))
        print('rank',r,'train loss',round(l,3),'fit',np.mean([x['fit'] for x in res]),'test',np.mean([x['test'] for x in res]),flush=True)
