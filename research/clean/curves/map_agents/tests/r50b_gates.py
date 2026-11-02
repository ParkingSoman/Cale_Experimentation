# serves: H-map
import numpy as np, json
H=np.load("cache/map/resnet50_h.npy").astype(np.float64)
sets={"pale":[734,342,1435,214],"lattice":[1852,1994],"flower_insect":[432,911,898,617]}
for k,u in sets.items():
    r=H[:,u].sum(1)
    print(k,[round(float(np.quantile(r,q)),2) for q in (.5,.8,.9,.95)], "other-channel mean act", round(float(H.mean()),3))
