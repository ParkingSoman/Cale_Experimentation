# serves: DQ19
from lib import *
from fit import *
t=ward(W); r,wA,lc,size=fitA(t)
print('in-sample A rms',float(((wA*r**2).sum()/wA.sum()).sqrt()))
# by sbar mass bin
s=wA
for lo,hi in [(0,1e-5),(1e-5,1e-4),(1e-4,1e-3),(1e-3,1e-2),(1e-2,1)]:
    mk=(s>=lo)&(s<hi); mass=float(s[mk].sum()/s.sum()); rms=float(((s[mk]*r[mk]**2).sum()/s[mk].sum()).sqrt())
    print(f'sbar in [{lo},{hi}) mass share {mass:.3f} rms {rms:.3f} count {int(mk.sum())}')
# by lca subtree size
sz=torch.from_numpy(size)[lc.clamp_min(0)]
for lo,hi in [(2,3),(3,6),(6,20),(20,100),(100,500),(500,1001)]:
    mk=(sz>=lo)&(sz<hi); 
    print(f'lca size [{lo},{hi}) mass share {float(s[mk].sum()/s.sum()):.3f} rms {float(((s[mk]*r[mk]**2).sum()/s[mk].sum()).sqrt()):.3f}')
