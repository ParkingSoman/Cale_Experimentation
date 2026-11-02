# serves: DQ11
"""Reads the least found on a symmetric table, as a network: DQ11's frame was beaten, so what does the better build do? K = 6, k = 3, a = 0.9. Prints the matrix of centred logits G (rows: the region,
columns: classes), its singular values, the student's confidence, and the value on the true softmax per row."""
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)
torch.manual_seed(3)
np.set_printoptions(precision=2, suppress=True, linewidth=150)
def run(K,k,ac):
    b=(1-ac)/(K-1)
    sb=torch.full((K,K),b)+torch.eye(K)*(ac-b)
    best=(1e9,None)
    for j in range(12):
        V=torch.randn(K,k).requires_grad_(True); c=torch.randn(K,k).requires_grad_(True); t=torch.zeros(K).requires_grad_(True)
        opt=torch.optim.LBFGS([V,c,t],lr=1,max_iter=3000,history_size=60,line_search_fn="strong_wolfe",tolerance_grad=1e-11,tolerance_change=1e-16)
        f=lambda:(sb*(sb.log()-torch.log_softmax(t+c@V.T,1))).sum(1).mean()
        def cl():
            opt.zero_grad(); v=f(); v.backward(); return v
        opt.step(cl)
        if float(f())<best[0]: best=(float(f()),(t+c@V.T).detach())
    L=best[1]; L=L-L.mean(1,keepdim=True)
    print(f"K={K} k={k} a={ac}: least found {best[0]:.6f}; rank bound: k + (t gives 1 more)")
    print("singular values",torch.linalg.svdvals(L).numpy())
    q=torch.softmax(L,1)
    print("student probability on own class per row",q.diag().numpy())
    print("KL per row",(sb*(sb.log()-q.log())).sum(1).numpy())
    print("student rows (probabilities)"); print(q.numpy())
    print("centred logits"); print(L.numpy())
    # order rows by the off-diagonal pattern
    print("off-diagonal sorted, row 0",np.sort(np.delete(q[0].numpy(),0)))
run(6,3,.9); run(6,2,.9); run(4,1,.9)
