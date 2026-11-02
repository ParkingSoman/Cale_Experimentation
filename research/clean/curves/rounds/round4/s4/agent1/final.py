# serves: DQ19
from lib import *
from scipy.spatial.distance import squareform
import json
Lg=H@W.T+b; T=1.5
P=np.exp((Lg-Lg.max(1,keepdims=True))/T); P/=P.sum(1,keepdims=True)
Ps=(P+P.T)/2; np.fill_diagonal(Ps,0)
D=1-Ps/Ps.max(); np.fill_diagonal(D,0)
t=nested(linkage(squareform(D,checks=False),'average'))
json.dump(t,open(HERE+'/tree_resnet18.json','w'))
print('A rms',round(score(t,'final'),4))
# channel reading: how many channels carry the logit of mass-bearing pairs
names=json.load(open(HERE+'/names.json'))
Pm=np.exp(Lg-Lg.max(1,keepdims=True)); Pm/=Pm.sum(1,keepdims=True); np.fill_diagonal(Pm,0)
pairs=np.dstack(np.unravel_index(np.argsort(-Pm.ravel())[:3000],Pm.shape))[0]
need=[];shared=[]
for c,i in pairs:
    contrib=H[c]*W[i]; pos=np.sort(contrib[contrib>0])[::-1]; cs=np.cumsum(pos)/pos.sum()
    need.append(int((cs<0.5).sum())+1)
print('top3000 mass pairs: channels needed for half the positive logit: median',np.median(need),'of 512; positives per pair median',int(np.median([ (H[c]*W[i]>0).sum() for c,i in pairs])))
# class-level: channels in common among each class's top-20 positive-W channels, for near siblings vs random
topk=lambda i: set(np.argsort(-W[i])[:20])
sib=[len(topk(c)&topk(i)) for c,i in pairs[:1000]]; rnd=[len(topk(c)&topk(i)) for c,i in np.random.default_rng(0).integers(0,1000,(1000,2))]
print('shared top-20 write/read channels: mass pairs',np.mean(sib),'random',np.mean(rnd))
