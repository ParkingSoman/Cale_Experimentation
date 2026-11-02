# serves: DQ19
import numpy as np, torchvision
from scipy.cluster.hierarchy import fcluster
from lib import *
names=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
m=np.load('cache/map/resnet18_meta.npz'); W=m['W']; lab=m['labels']
h=np.load('cache/map/resnet18_h.npy'); hA=h[np.isin(lab,A)]
Z=linkage(W,'ward'); rng=np.random.default_rng(0)
for k in (6,):
    cl=fcluster(Z,k,'maxclust')
    for g in range(1,k+1):
        mem=np.where(cl==g)[0]; 
        ma=[c for c in mem if c in set(A)]
        # strong-firing channels on photos of member classes in A
        ph=np.isin(lab,ma); fr=(h[ph]>3).mean(0)-(hA>3).mean(0)
        print(g,len(mem),'e.g.',', '.join(names[i] for i in rng.choice(mem,10,replace=False)),'| over-firing channels',np.argsort(-fr)[:5].tolist(),np.round(np.sort(-fr)[:5]*-1,2).tolist())
