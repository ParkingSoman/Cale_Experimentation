# serves: DQ19
import numpy as np, torch, torchvision
names=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
D='cache/map/'
y=np.load(D+'resnet18_y.npy'); m=np.load(D+'resnet18_meta.npz'); lab=m['labels']
S=torch.softmax(torch.from_numpy(y).double(),1).numpy()
sb=np.stack([S[lab==c].mean(0) for c in range(1000)])
np.save('rounds/round4/s4/agent2/sb.npy',sb)
perm=np.random.default_rng(2026).permutation(1000); A=np.sort(perm[:500]); np.save('rounds/round4/s4/agent2/A.npy',A)
off=sb.copy(); np.fill_diagonal(off,0)
print('diag mean',np.diag(sb).mean(),'offdiag mass per row mean',off.sum(1).mean())
rng=np.random.default_rng(0)
for c in rng.choice(A,14,replace=False):
    t=np.argsort(-off[c])[:6]
    print(f'{c} {names[c]} diag={sb[c,c]:.2f} | '+'; '.join(f'{names[i]} {off[c,i]:.3f}' for i in t))
# entropy of offmass: how concentrated
p=off/off.sum(1,keepdims=True); ent=-(p*np.log(p+1e-30)).sum(1); print('median off-mass entropy (nats)',np.median(ent[A]),'max',np.log(999))
cum=np.sort(p,1)[:,::-1].cumsum(1); print('median classes for 50/80/95% of offmass',[int(np.median((cum[A]<q).sum(1)))+1 for q in (.5,.8,.95)])
