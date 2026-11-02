# serves: H-map
import numpy as np
Z = np.load('cache/map/resnet18_z.npy', mmap_mode='r')
U = [147, 153, 161, 179, 221, 327, 366, 371, 416]
z = np.asarray(Z[:, U], float)
c = np.corrcoef(z.T)
print('pre-activation corr over P')
print('     ' + ' '.join(f'{u:5d}' for u in U))
for i, u in enumerate(U):
    print(f'{u:4d} ' + ' '.join(f'{c[i, j]:5.2f}' for j in range(len(U))))
