# serves: DQ19
from lib import *
from scipy.cluster.hierarchy import linkage, to_tree
Z=linkage(W,'ward')
root=to_tree(Z)
def build(n,minsize):
    # collapse subtrees smaller than minsize into flat lists
    if n.is_leaf(): return n.id
    if n.count<=minsize: return [i for i in n.pre_order()]
    return [build(n.left,minsize),build(n.right,minsize)]
for ms in [1,2,3,4,6,8,12,20,40]:
    t=build(root,ms); print('collapse<=',ms,round(score(t,'e5'),4))
