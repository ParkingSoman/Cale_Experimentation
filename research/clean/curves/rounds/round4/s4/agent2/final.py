# serves: DQ19
import numpy as np, json, sys
from lib import *
from scipy.spatial.distance import pdist,squareform
m=np.load('cache/map/resnet18_meta.npz'); W=m['W']
Dw=squareform(pdist(W)); Dw/=np.median(np.sort(Dw,1)[:,1])
def evid(rows):
    E=np.zeros((C,C)); E[rows]=sb[rows]; np.fill_diagonal(E,0); return np.maximum(E,E.T)
def linkZ(rows,rho,sat,meth='average'):
    E=evid(rows); D=Dw*(1-rho*np.minimum(1,E/sat)); np.fill_diagonal(D,0); D=(D+D.T)/2
    return linkage(squareform(D,checks=False),meth)
def flatten(Z,thr):
    n=1000; nodes=[None]*(2*n-1)
    for i in range(n): nodes[i]=(i,0.0,True)
    for k,(x,y,hh,_) in enumerate(Z):
        ch=[]
        for u in (nodes[int(x)],nodes[int(y)]):
            if (not u[2]) and hh-u[1]<thr*hh: ch+=u[0]
            else: ch.append(u[0])
        nodes[n+k]=(ch,hh,False)
    def conv(u): return [conv(v) for v in u] if isinstance(u,list) else u
    return conv(nodes[-1][0])
if __name__=='__main__':
    sp=[splits(1),splits(2)]
    def ev(name,mk):
        r=[pseudo(mk(a),a,b) for a,b in sp]
        print(f"{name:30s} fit {np.mean([x['fit'] for x in r]):.3f} test {np.mean([x['test'] for x in r]):.3f}",flush=True)
    for thr in (0,0.1,0.2,0.3):
        ev(f'hyb .6/.1 flat{thr}',lambda a:flatten(linkZ(a,0.6,0.1),thr))
    for thr in (0.1,0.2): ev(f'W avg flat{thr}',lambda a:flatten(linkZ(a,0,1),thr))
    if len(sys.argv)>1:
        thr=float(sys.argv[1]); t=flatten(linkZ(A,0.6,0.1),thr); json.dump(t,open(R+'tree_resnet18.json','w'))
