# serves: DQ19
import numpy as np
from lib import *
from scipy.spatial.distance import pdist,squareform
m=np.load('cache/map/resnet18_meta.npz'); W=m['W']
Dw=squareform(pdist(W)); Dw/=np.median(np.sort(Dw,1)[:,1])
sp=[splits(1),splits(2)]
def evid(rows):
    E=np.zeros((C,C)); E[rows]=sb[rows]; np.fill_diagonal(E,0)
    return np.maximum(E,E.T)
def hyb(rows,rho,sat,meth='average'):
    E=evid(rows); D=Dw*(1-rho*np.minimum(1,E/sat)); np.fill_diagonal(D,0)
    D=(D+D.T)/2
    from scipy.spatial.distance import squareform as sf
    return nest(linkage(sf(D,checks=False),meth))
def ev(name,mk):
    r=[pseudo(mk(a),a,b) for a,b in sp]
    print(f"{name:30s} fit {np.mean([x['fit'] for x in r]):.3f} test {np.mean([x['test'] for x in r]):.3f}",flush=True)
ev('average W only',lambda a:hyb(a,0,1))
for rho in (0.3,0.6,0.85):
    for sat in (0.03,0.1):
        ev(f'avg rho{rho} sat{sat}',lambda a:hyb(a,rho,sat))
ev('ward-like: centroid? skip',lambda a:hyb(a,0.6,0.1,'complete'))
