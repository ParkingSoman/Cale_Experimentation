# serves: H-map
import numpy as np,sys
H=np.load('cache/map/resnet50_h.npy',mmap_mode='r')
us=[int(x) for x in sys.argv[1].split(',')]
X=np.stack([np.asarray(H[:,u],dtype=np.float64) for u in us],1)
print('corr\n',np.round(np.corrcoef(X.T),2))
for i,u in enumerate(us):
    top=X[:,i]>np.quantile(X[:,i],0.98)
    print(u,'on top2% means of others',np.round(X[top].mean(0),2),'overall',np.round(X.mean(0),2))
