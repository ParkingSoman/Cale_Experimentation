# serves: DQ19
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster
D='cache/map/'
h=np.load(D+'resnet18_h.npy'); m=np.load(D+'resnet18_meta.npz'); W=m['W']; lab=m['labels']
hc=h-h.mean(0); sd=hc.std(0)+1e-9
C=(hc/sd).T@(hc/sd)/len(h)
print('offdiag corr abs mean',np.abs(C-np.eye(512)).mean(), 'frac>0.3',(np.abs(C-np.eye(512))>0.3).mean())
# strong co-firing
S=(h>2).astype(float); print('strong rate per channel mean',S.mean())
d=1-C; iu=np.triu_indices(512,1)
Z=linkage(d[iu],'average')
for k in (4,8,16,32):
    cl=fcluster(Z,k,'maxclust'); print(k,np.bincount(cl)[1:])
np.save('rounds/round4/s4/agent2/chan_corr.npy',C)
