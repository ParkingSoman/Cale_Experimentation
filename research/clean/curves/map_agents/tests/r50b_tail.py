# serves: H-map
import numpy as np
H=np.load("cache/map/resnet50_h.npy").astype(np.float64)
m=np.load("cache/map/resnet50_meta.npz");W=m["W"].astype(np.float64);Wc=W-W.mean(0)
allc=H.sum(0)
print("typical channel: share of activity in top 5% / 20% photos")
def sh(u):
    h=np.sort(H[:,u])[::-1];c=h.cumsum()/h.sum();n=len(h);return c[int(.05*n)],c[int(.2*n)]
s=np.array([sh(u) for u in range(0,2048,16)]);print("median over 128 chans",np.median(s,0).round(2))
for u in [432,544,734,969,1350,1852,1882,1895,342,911,1994]:
    print(u,np.round(sh(u),2))
# logit write norm per photo vs class: how spread is channel's write across top-1 class
