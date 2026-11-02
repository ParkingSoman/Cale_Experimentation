# serves: H-map
import numpy as np
M=np.load("cache/map/resnet18_meta.npz"); ph=M["photos"]
X=np.load("../hf/imagenet/val_224.npy",mmap_mode="r")
H=np.load("cache/map/resnet18_h.npy",mmap_mode="r")
F=[]
for i in ph:
    x=np.asarray(X[i])[::4,::4].astype(np.float32); g=x.mean(-1)
    F.append([x[...,0].mean()-x[...,2].mean(), g.mean(), (x.max(-1)-x.min(-1)).mean(), (g<40).mean(), g.std(), (g>200).mean()])
F=np.array(F); names=["warm R-B","lum","sat","darkfrac","contrast","brightfrac"]
for u in [120,222,224,273,401,404]:
    h=np.asarray(H[:,u]); h=h.mean(tuple(range(1,h.ndim))) if h.ndim>1 else h
    print(u,{k:round(float(np.corrcoef(h,F[:,j])[0,1]),2) for j,k in enumerate(names)})
