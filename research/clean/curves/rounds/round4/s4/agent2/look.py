# serves: DQ19
import numpy as np
D='cache/map/'
z=np.load(D+'resnet18_z.npy',mmap_mode='r'); h=np.load(D+'resnet18_h.npy',mmap_mode='r'); y=np.load(D+'resnet18_y.npy',mmap_mode='r')
m=np.load(D+'resnet18_meta.npz'); print(z.shape,h.shape,y.shape,{k:m[k].shape for k in m})
h=np.asarray(h); z=np.asarray(z)
print('h==relu(z)?',np.abs(h-np.maximum(z,0)).max(), 'h range',h.min(),h.max(),h.mean())
print('frac h>1',(h>1).mean(),'>3',(h>3).mean(), 'mean strong per photo',(h>3).sum(1).mean())
W=m['W'];b=m['b'];print(np.abs(y-(h@W.T+b)).max() if True else 0)
