# serves: DQ19
import numpy as np, torch, torchvision
from lib import *
import lib
names=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
m=np.load('cache/map/resnet18_meta.npz'); W=m['W']
tree=nest(linkage(W,'ward')); A1,A2=splits(1)
torch.set_default_dtype(torch.float64)
s=torch.from_numpy(sb).clamp_min(1e-12); s=s/s.sum(1,keepdim=True); L=s.log()
lca,nn=lca_of(tree); off=~torch.eye(C,dtype=torch.bool); w=s*off
F=torch.from_numpy(A1); T=torch.from_numpy(A2)
beta=torch.zeros(C); fv=torch.zeros(C,C); wA,LA=w[F],L[F]; idxA=lca[F].reshape(-1); idx=lca.reshape(-1)
for _ in range(40):
    alpha=(wA*(LA-beta[None]-fv[F])).sum(1)/wA.sum(1)
    beta=(wA*(LA-alpha[:,None]-fv[F])).sum(0)/wA.sum(0).clamp_min(1e-300)
    r=LA-alpha[:,None]-beta[None]
    num=torch.zeros(nn).index_add_(0,idxA,(wA*r).reshape(-1)); den=torch.zeros(nn).index_add_(0,idxA,wA.reshape(-1))
    f=num/den.clamp_min(1e-300); fv=f[idx].reshape(C,C)
wR,LR=w[T],L[T]; al=(wR*(LR-beta[None]-fv[T])).sum(1)/wR.sum(1); res=LR-al[:,None]-beta[None]-fv[T]
sq=(wR*res**2); tot=sq.sum()
# by rank of entry within row mass
rk=torch.argsort(torch.argsort(-wR,1),1)
for lo,hi in ((0,1),(1,3),(3,10),(10,50),(50,1000)):
    mk=(rk>=lo)&(rk<hi); print(f'rank {lo}-{hi}: share of weight {wR[mk].sum()/wR.sum():.3f}, share of sq err {sq[mk].sum()/tot:.3f}, rms {(sq[mk].sum()/wR[mk].sum()).sqrt():.2f}, mean resid {((wR*res)[mk].sum()/wR[mk].sum()):.2f}')
# by lca size: how big is the smallest node containing c and i
sizes=np.zeros(nn,int)
def cnt(n):
    if isinstance(n,int): return 1
    return sum(cnt(x) for x in n)
# worst rows
rowerr=sq.sum(1)/wR.sum(1)
for j in torch.argsort(-sq.sum(1))[:6]:
    c=int(T[j]); k=torch.argsort(-sq[j])[:3]
    print(names[c],'|',[(names[int(i)],round(float(res[j,i]),1),round(float(wR[j,i]),3)) for i in k])
np.save(R+'resid_test.npy',res.numpy())
