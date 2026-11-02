# serves: DQ19
from lib import *
import torchvision
names=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
import json; json.dump(names,open(HERE+'/names.json','w'))
from e2 import bisect
t=bisect(np.arange(1000),W)
def leaves(n): return [n] if isinstance(n,int) else [x for c in n for x in leaves(c)]
def show(n,depth,maxd=3,pre=''):
    L=np.array(leaves(n))
    if len(L)<2: return
    # channel selectivity: mean W of this set minus overall
    if depth>=maxd: return
    a,b_=n
    La,Lb=np.array(leaves(a)),np.array(leaves(b_))
    diff=W[La].mean(0)-W[Lb].mean(0); top=np.argsort(-np.abs(diff))[:6]
    print(pre,len(La),'vs',len(Lb),'chans',[(int(k),round(float(diff[k]),2)) for k in top])
    print(pre,' A:',[names[i] for i in La[:8]],' B:',[names[i] for i in Lb[:8]])
    show(a,depth+1,maxd,pre+'  A'); show(b_,depth+1,maxd,pre+'  B')
show(t,0)
