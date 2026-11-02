# serves: DQ19
from lib import *
from scipy.spatial.distance import squareform, pdist
Lg=H@W.T+b
P=np.exp(Lg-Lg.max(1,keepdims=True)); P/=P.sum(1,keepdims=True)
def aff(P,sym='mean'):
    Ps=(P+P.T)/2 if sym=='mean' else (np.maximum(P,P.T) if sym=='max' else np.sqrt(P*P.T))
    np.fill_diagonal(Ps,0); d=Ps.sum(1); return Ps/np.sqrt(np.outer(d,d))
def tr(A_,meth='average'):
    D=1-A_/A_.max(); np.fill_diagonal(D,0); return nested(linkage(squareform(D,checks=False),meth))
for sym in ['mean','max','geo']:
    A_=aff(P,sym); print(sym,round(score(tr(A_),'e9'),4))
A_=aff(P)
for g in [0.5,0.75,1.5]:
    print('pow',g,round(score(tr(A_**g),'e9'),4))
# temperature: sharpen/flatten wiring logits
for T in [0.5,0.75,1.5]:
    Pt=np.exp((Lg-Lg.max(1,keepdims=True))/T); Pt/=Pt.sum(1,keepdims=True); print('T',T,round(score(tr(aff(Pt)),'e9'),4))
# no row/col normalisation
Ps=(P+P.T)/2; np.fill_diagonal(Ps,0); print('raw',round(score(tr(Ps),'e9'),4))
# remove column scale (beta-like): double normalise by iterating Sinkhorn
S=Ps.copy()
for _ in range(20): S/=S.sum(1,keepdims=True); S/=S.sum(0,keepdims=True); S=(S+S.T)/2
print('sinkhorn',round(score(tr(S),'e9'),4))
