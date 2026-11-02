# serves: DQ19
from lib import *
import sys
def bisect(ids,X,minsize=2,it=20):
    if len(ids)==1: return int(ids[0])
    if len(ids)==2: return [int(ids[0]),int(ids[1])]
    Xi=X[ids]; Xc=Xi-Xi.mean(0)
    u,s,vt=np.linalg.svd(Xc,full_matrices=False); p=Xc@vt[0]
    lab_=p>0
    # 2-means refine
    c=np.array([Xi[~lab_].mean(0),Xi[lab_].mean(0)]) if lab_.any() and (~lab_).any() else None
    if c is None: lab_=p>np.median(p); c=np.array([Xi[~lab_].mean(0),Xi[lab_].mean(0)])
    for _ in range(it):
        d=((Xi[:,None,:]-c[None])**2).sum(2); nl=d[:,1]<d[:,0]
        if nl.all() or (~nl).all(): break
        c=np.array([Xi[~nl].mean(0),Xi[nl].mean(0)]); lab_=nl
    if lab_.all() or (~lab_).all(): lab_=p>np.median(p)
    return [bisect(ids[~lab_],X),bisect(ids[lab_],X)]
if __name__=='__main__':
    ids=np.arange(1000)
    for nm,X in [('W',W),('W+',np.maximum(W,0))]:
        t=bisect(ids,X); print(nm,round(score(t,'e2'),4))
