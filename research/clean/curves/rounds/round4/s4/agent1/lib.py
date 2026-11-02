# serves: DQ19
import numpy as np, json, subprocess, os
from scipy.cluster.hierarchy import linkage
R='/Users/shonusengupta/Cale_Experimentation/research/clean/curves/'
HERE=os.path.dirname(os.path.abspath(__file__))
m=np.load(R+'cache/map/resnet18_meta.npz'); W=m['W'].astype(np.float64); b=m['b']; lab=m['labels']
h=np.load(R+'cache/map/resnet18_h.npy').astype(np.float64)
H=np.stack([h[lab==c].mean(0) for c in range(1000)])
def nested(Z,n=1000):
    nodes=list(range(n))
    for x,y,_,_ in Z: nodes.append([nodes[int(x)],nodes[int(y)]])
    return nodes[-1]
def score(tree,name='t',net='resnet18'):
    p=f'{HERE}/tmp_{name}.json'; json.dump(tree,open(p,'w'))
    o=subprocess.run([R+'.venv/bin/python',R+'pc_r4_s4_eval.py','--net',net,'--tree',p,'--split','A'],capture_output=True,text=True,cwd=R)
    return json.loads(o.stdout)['rms']
def ward(X,method='ward',metric='euclidean'):
    return nested(linkage(X,method,metric=metric))
