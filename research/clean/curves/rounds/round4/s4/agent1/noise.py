# serves: DQ19
import numpy as np, torch
R='/Users/shonusengupta/Cale_Experimentation/research/clean/curves/'
Y=np.load(R+'cache/map/resnet18_y.npy'); lab=np.load(R+'cache/map/resnet18_meta.npz')['labels']
S=torch.softmax(torch.from_numpy(Y.astype(np.float64)),1)
K=1000
perm=np.random.default_rng(2026).permutation(K); A=perm[:500]
s1=[];s2=[];sf=[]
for c in A:
    ix=np.where(lab==c)[0]; 
    s1.append(S[ix[::2]].mean(0)); s2.append(S[ix[1::2]].mean(0)); sf.append(S[ix].mean(0))
s1=torch.stack(s1);s2=torch.stack(s2);sf=torch.stack(sf)
off=torch.ones(500,1000,dtype=torch.bool)
for r,c in enumerate(A): off[r,c]=False
w=sf*off
d=(s1.clamp_min(1e-12).log()-s2.clamp_min(1e-12).log())
# remove row scale
wr=w; al=(wr*d).sum(1)/wr.sum(1)
d=d-al[:,None]
print('rms diff/sqrt2 (weights sbar):',float(((w*d**2).sum()/w.sum()).sqrt()/2**.5))
# mass structure
print('row mass off-diag mean',float(w.sum(1).mean()))
srt=torch.sort(w,1,descending=True).values
cum=srt.cumsum(1)/w.sum(1,keepdim=True)
print('entries to hold 50/80/95% of off-diag mass (median):',[int((cum<q).sum(1).float().median())+1 for q in (.5,.8,.95)])
