# serves: DQ19
import numpy as np
from lib import *
from scipy.cluster.hierarchy import cut_tree
D='cache/map/'
m=np.load(D+'resnet18_meta.npz'); W=m['W']; b=m['b']
sp=[splits(1),splits(2)]
def ev(name,t):
    r=[pseudo(t,a,b_) for a,b_ in sp]
    print(f"{name:28s} fit {np.mean([x['fit'] for x in r]):.3f} test {np.mean([x['test'] for x in r]):.3f}",flush=True)
rng=np.random.default_rng(19)
Z=linkage(W,'ward'); ev('ward W',nest(Z))
ev('shuffled',nest(Z,1000) if False else nest(Z))  # placeholder replaced below
def shuf():
    nodes=[int(x) for x in rng.permutation(1000)]
    for x,y,_,_ in Z: nodes.append([nodes[int(x)],nodes[int(y)]])
    return nodes[-1]
ev('shuffled W-ward',shuf())
Wn=W/np.linalg.norm(W,axis=1,keepdims=True)
ev('ward Wcos',nest(linkage(Wn,'ward')))
ev('average Wcos',nest(linkage(Wn,'average',metric='cosine') if False else linkage(Wn,'average')))
ev('complete W',nest(linkage(W,'complete')))
Wb=np.c_[W,b[:,None]*3]; ev('ward W+b',nest(linkage(Wb,'ward')))
# flatten: cut ward tree into k clusters, then ward inside? use k-level multi-fanout: merge nodes with height< q
def flatten(Z,thr):
    n=1000; kids={}; hts={}
    nodes=[None]*(2*n-1)
    for i in range(n): nodes[i]=i
    for k,(x,y,hh,_) in enumerate(Z):
        a,b_=nodes[int(x)],nodes[int(y)]
        ch=[]
        for u in (a,b_):
            if isinstance(u,tuple) and hh-u[1]<thr*hh: ch+=u[0]   # absorb child whose height close to parent
            else: ch.append(u[0] if isinstance(u,tuple) and False else (u if not isinstance(u,tuple) else u[0]))
        nodes[n+k]=(ch,hh)
    root=nodes[-1]
    def conv(u):
        if isinstance(u,tuple): return [conv(v) for v in u[0]]
        return u
    return conv(root)
for thr in (0.03,0.08,0.15):
    t=flatten(Z,thr); ev(f'flatten {thr}',t)
