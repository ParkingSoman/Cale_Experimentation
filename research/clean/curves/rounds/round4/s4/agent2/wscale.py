# serves: DQ19
import numpy as np, json, subprocess
from scipy.cluster.hierarchy import linkage
D='cache/map/'
A=np.load('rounds/round4/s4/agent2/A.npy')
h=np.load(D+'resnet18_h.npy'); m=np.load(D+'resnet18_meta.npz'); W=m['W']; lab=m['labels']
hA=h[np.isin(lab,A)]    # reading channels on A-class photos only
def nest(Z,n=1000):
    nodes=list(range(n))
    for x,y,_,_ in Z: nodes.append([nodes[int(x)],nodes[int(y)]])
    return nodes[-1]
def score(name,X,method='ward'):
    p=f'rounds/round4/s4/agent2/{name}.json'; json.dump(nest(linkage(X,method)),open(p,'w'))
    out=subprocess.run(['.venv/bin/python','pc_r4_s4_eval.py','--net','resnet18','--tree',p,'--split','A'],capture_output=True,text=True).stdout
    print(name,json.loads(out)['rms'],flush=True)
sd=hA.std(0); mu=hA.mean(0); rms=np.sqrt((hA**2).mean(0))
Wc=W-W.mean(0)
score('w_plain',W); score('w_centered',Wc)
score('w_sd',W*sd); score('w_mu',W*mu); score('w_rms',W*rms); score('wc_sd',Wc*sd); score('wc_sd2',Wc*sd**2)
for p in (0.5,1.5): score(f'wc_sd{p}',Wc*sd**p)
