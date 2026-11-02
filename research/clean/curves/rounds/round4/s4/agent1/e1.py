# serves: DQ19
from lib import *
hm=H.mean(0)
def z(X): return X/np.linalg.norm(X,axis=1,keepdims=True)
Hc=H-H.mean(0)
V={'W':W,'W*sqrt hm':W*np.sqrt(hm),'W+':np.maximum(W,0),'H':H,'Hc':Hc,'Hc/std':Hc/Hc.std(0),'W|Hc':np.hstack([z(W),z(Hc)]),
 'W*Hstd':W*H.std(0),'z(W)':z(W)}
for k,X in V.items(): print(k,round(score(ward(X),'e1'),4))
