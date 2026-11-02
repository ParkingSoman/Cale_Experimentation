# serves: DQ19
from lib import *
hc=h-h.mean(0)
cov=np.cov(hc.T); ev,U=np.linalg.eigh(cov); Sg=U*np.sqrt(np.maximum(ev,0))
# between-class cov only (what separates classes)
Hc=H-H.mean(0); cb=np.cov(Hc.T); ev2,U2=np.linalg.eigh(cb); Sb=U2*np.sqrt(np.maximum(ev2,0))
def z(X): return X/np.linalg.norm(X,axis=1,keepdims=True)
Lg=H@W.T+b   # writer c, reader i
Lc=Lg-Lg.mean(1,keepdims=True)
P=np.exp(Lg-Lg.max(1,keepdims=True)); P/=P.sum(1,keepdims=True); LP=np.log(P+1e-12)
res={}
res['W cov_all']=ward(W@Sg)
res['W cov_between']=ward(W@Sb)
res['Lg rows']=ward(Lc)
res['Lg cols']=ward(Lg.T-Lg.T.mean(1,keepdims=True))
res['rows|cols']=ward(np.hstack([z(Lc),z(Lg.T-Lg.T.mean(1,keepdims=True))]))
res['sqrtP rows|cols']=ward(np.hstack([np.sqrt(P),np.sqrt(P).T]))
for k,t in res.items(): print(k,round(score(t,'e4'),4))
