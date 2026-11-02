# serves: H-map
import numpy as np
H=np.load("cache/map/resnet18_h.npy",mmap_mode="r")
u=[120,147,222,224,273,401,404]
A=np.asarray(H[:,u]); print(u); print(np.round(np.corrcoef(A.T),2))
