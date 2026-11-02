# serves: DQ19
import numpy as np, json
from lib import *
D='cache/map/'
m=np.load(D+'resnet18_meta.npz'); W=m['W']
A1,A2=splits(1)
def feat_cols(rows,kind):
    s=np.clip(sb[rows],1e-12,None); s=s/s.sum(1,keepdims=True); L=np.log(s)
    X=L.copy()
    for r,c in enumerate(rows): X[r,c]=np.nan
    # remove row scale using weighted off-diag mean
    w=s.copy(); 
    for r,c in enumerate(rows): w[r,c]=0
    al=np.nansum(np.where(np.isnan(X),0,w*X),1)/w.sum(1); X=X-al[:,None]
    # diag fill with row's column mean
    fill=np.nanmean(X,1)
    X=np.where(np.isnan(X),fill[:,None],X)
    if kind=='log': return X.T
    if kind=='logw': return (X*np.sqrt(w+1e-4)).T   # mass-aware
    if kind=='sqrt': 
        P=np.sqrt(s); P=P/np.linalg.norm(P,axis=1,keepdims=True); return P.T
    if kind=='prob':
        P=s.copy(); 
        for r,c in enumerate(rows): P[r,c]=0
        return P.T
res={}
print('Ward on W:',pseudo(nest(linkage(W,'ward')),A1,A2))
for kind in ('log','logw','sqrt','prob'):
    X=feat_cols(A1,kind)
    for meth in ('ward','average'):
        t=nest(linkage(X,meth)); print(kind,meth,pseudo(t,A1,A2),flush=True)
