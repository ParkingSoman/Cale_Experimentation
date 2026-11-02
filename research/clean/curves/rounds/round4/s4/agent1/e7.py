# serves: DQ19
from lib import *
from fit import *
import json
names=json.load(open(HERE+'/names.json'))
t=ward(W); r,wA,lc,size=fitA(t)
sz=torch.from_numpy(size)[lc.clamp_min(0)]
mk=(wA>0.02)&(sz>=100)
print('big-mass pairs',int(mk.sum()),'mass share',float(wA[mk].sum()/wA.sum()))
Aidx=A.numpy()
ii=torch.nonzero(mk)
order=torch.argsort(-wA[mk])[:45]
for o in order:
    a,i=ii[o].tolist(); c=Aidx[a]
    print(f'{names[c][:22]:22s} -> {names[i][:22]:22s} sbar {float(wA[a,i]):.3f} lcasize {int(sz[a,i])} resid {float(r[a,i]):+.2f}')
