# serves: DQ19
import numpy as np, torch
m=np.load('cache/map/resnet18_meta.npz'); lab=m['labels']; y=np.load('cache/map/resnet18_y.npy')
A=np.load('rounds/round4/s4/agent2/A.npy')
S=torch.softmax(torch.from_numpy(y).double(),1).numpy()
rng=np.random.default_rng(5); num=0;den=0;num2=0
for c in A:
    idx=np.where(lab==c)[0]; p=rng.permutation(idx); a=S[p[:7]].mean(0); b=S[p[7:14]].mean(0); f=S[p[14:]].mean(0)
    w=f.copy(); w[c]=0; w/=w.sum()
    la,lb=np.log(np.clip(a,1e-12,None)),np.log(np.clip(b,1e-12,None)); d=la-lb; d=d-(w*d).sum()  # row scale free
    d[c]=0
    num+=(w*d**2).sum(); den+=1
print('rms of (log half1 - log half2), row-scale removed:',np.sqrt(num/den),' => noise of 20-photo mean approx (7-photo halves, scaled by sqrt(7/20)/sqrt2)',np.sqrt(num/den)/np.sqrt(2)*np.sqrt(7/20))
