# serves: DQ19
import numpy as np
R='/Users/shonusengupta/Cale_Experimentation/research/clean/curves/'
m=np.load(R+'cache/map/resnet18_meta.npz'); W=m['W'];b=m['b'];lab=m['labels']
h=np.load(R+'cache/map/resnet18_h.npy',mmap_mode='r'); print(W.shape,h.shape,b[:3])
h=np.asarray(h,dtype=np.float32)
print('h mean',h.mean(),'max',h.max(),'frac>0',(h>0).mean())
hm=h.mean(0); print('chan mean h: min,med,max',hm.min(),np.median(hm),hm.max())
print('W row mean',W.mean(), 'std',W.std(),'col std range',W.std(0).min(),W.std(0).max())
# per-channel positive share
pos=(W>0).mean(0); print('frac positive per chan',pos.min(),np.median(pos),pos.max())
u,s,vt=np.linalg.svd(W-W.mean(0),full_matrices=False); print('sv',s[:15].round(1), (s**2/ (s**2).sum())[:10].round(3))
