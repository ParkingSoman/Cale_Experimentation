# serves: DQ19
from lib import *
from scipy.spatial.distance import squareform
def z(X): return X/np.linalg.norm(X,axis=1,keepdims=True)
Hc=H-H.mean(0)
Lg=H@W.T+b
P=np.exp(Lg-Lg.max(1,keepdims=True)); P/=P.sum(1,keepdims=True)
# photo-level wiring-predicted table is the true table; use class-mean version P (no table rows)
Ps=(P+P.T)/2; np.fill_diagonal(Ps,0)
# normalise row/col scale: divide by sqrt(rowmass colmass)
d=Ps.sum(1); Pn=Ps/np.sqrt(np.outer(d,d))
for meth in ['average','complete','weighted']:
    D=1-Pn/Pn.max(); np.fill_diagonal(D,0)
    t=nested(linkage(squareform(D,checks=False),meth)); print('graph',meth,round(score(t,'e8'),4))
for a_ in [0.5,1,2,4]:
    print('W|Hc weight',a_,round(score(ward(np.hstack([z(W),a_*z(Hc)])),'e8'),4))
# W plus graph embedding
u,s,vt=np.linalg.svd(Pn); E=u[:,:40]*np.sqrt(s[:40])
for k in [20,40]:
    E=u[:,:k]*np.sqrt(s[:k]); print('spectral',k,round(score(ward(E),'e8'),4), round(score(ward(np.hstack([z(W),z(E)])),'e8'),4))
