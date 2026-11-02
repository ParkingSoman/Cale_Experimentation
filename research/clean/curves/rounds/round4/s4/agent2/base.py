# serves: DQ19
import numpy as np, json, sys
from scipy.cluster.hierarchy import linkage
D='cache/map/'
h=np.load(D+'resnet18_h.npy'); m=np.load(D+'resnet18_meta.npz'); W=m['W']; lab=m['labels']
Hc=np.stack([h[lab==c].mean(0) for c in range(1000)])
np.save('rounds/round4/s4/agent2/Hc.npy',Hc)
def nest(Z,n=1000):
    nodes=list(range(n))
    for x,y,_,_ in Z: nodes.append([nodes[int(x)],nodes[int(y)]])
    return nodes[-1]
def save(name,X,method='ward'):
    json.dump(nest(linkage(X,method)),open(f'rounds/round4/s4/agent2/{name}.json','w'))
save('t_hc',Hc)
Hs=Hc-Hc.mean(0); save('t_hc_centered',Hs)
save('t_hcW',Hc@W.T)           # class mean logit-ish rows
save('t_hc_cos',Hs/np.linalg.norm(Hs,axis=1,keepdims=True))
