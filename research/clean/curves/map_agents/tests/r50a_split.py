# serves: H-map
import numpy as np,collections,torchvision
m=np.load('cache/map/resnet50_meta.npz');lab=m['labels']
H=np.load('cache/map/resnet50_h.npy',mmap_mode='r')
a=np.asarray(H[:,364],float);b=np.asarray(H[:,972],float)
nm=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta['categories']
top=a>np.quantile(a,0.97)
print('n',top.sum(),'frac 972 also high',(b[top]>np.quantile(b,0.97)).mean())
for nmx,s in(('364 hi, 972 low',top&(b<np.quantile(b,0.7))),('both',top&(b>=np.quantile(b,0.97)))):
    c=collections.Counter(nm[l] for l in lab[s]);print(nmx,s.sum(),c.most_common(12))
top2=b>np.quantile(b,0.97)
print('972 hi, 364 low',collections.Counter(nm[l] for l in lab[top2&(a<np.quantile(a,0.7))]).most_common(8), (top2&(a<np.quantile(a,0.7))).sum())
