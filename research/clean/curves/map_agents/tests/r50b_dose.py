# serves: H-map
import numpy as np
H=np.load("cache/map/resnet50_h.npy").astype(np.float64)
m=np.load("cache/map/resnet50_meta.npz");W=m["W"].astype(np.float64);lab=m["labels"]
Y=np.load("cache/map/resnet50_y.npy").astype(np.float64)
def rk(x): return np.argsort(np.argsort(x)).astype(float)
for u in [432,544,734,969,1350,1852,1882,1895,342,911,1994]:
    k=int(np.argmax(W[:,u])); h=H[:,u]; y=Y[:,k]-Y.mean(1)
    q=np.quantile(h,[.5,.8])
    out=[]
    for lo,hi in [(-1,q[0]),(q[0],q[1]),(q[1],1e9)]:
        s=(h>lo)&(h<=hi); out.append(round(np.corrcoef(rk(h[s]),rk(y[s]))[0,1],2))
    print(u,"top reader class",k,"rank-corr h vs centred logit: weak50%, mid30%, top20% =",out, " mean h on that class",round(h[lab==k].mean(),2),"vs elsewhere",round(h[lab!=k].mean(),2))
