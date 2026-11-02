# serves: H-map
import numpy as np
M=np.load("cache/map/resnet18_meta.npz"); W=M["W"].astype(np.float64)
H=np.load("cache/map/resnet18_h.npy").astype(np.float64); Y=np.load("cache/map/resnet18_y.npy").astype(np.float64)
n=len(H); idx=np.arange(0,n,10)
top=Y.argmax(1); Ys=np.sort(Y,1); sec=np.argsort(Y,1)[:,-2]
ks=[];ov=[]
mine=[120,222,224,273,401,404]
for i in idx:
    c=H[i]*(W[top[i]]-W[sec[i]])
    o=np.sort(c)[::-1]; pos=o[o>0]; cs=np.cumsum(pos)/pos.sum()
    ks.append((np.searchsorted(cs,0.5)+1, np.searchsorted(cs,0.9)+1))
    ov.append(c[mine].sum()/c.sum() if c.sum()!=0 else 0)
ks=np.array(ks); print("channels for 50%/90% of positive margin: median",np.median(ks,0),"of 512")
print("six channels' share of margin: mean",np.mean(ov),"max abs",np.max(np.abs(ov)))
# how much does one channel's share vary across photos: top-1 channel share
print("largest single channel share of positive margin, median:",np.median([ (lambda c:(c.max()/c[c>0].sum()))(H[i]*(W[top[i]]-W[sec[i]])) for i in idx]))
