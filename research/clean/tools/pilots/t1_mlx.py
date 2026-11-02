import sys, os, json
ROOT="/Users/shonusengupta/Cale_Experimentation"
sys.path.insert(0,ROOT+"/research/tree")
import mlx.core as mx
import tree_lib as T
model,tok=T.load_model()
blocks=T.token_blocks(tok,"test",4)
L=28
NS=list(range(300)) if os.environ.get("BIG") else [100]
b=T.eval_loss(model,blocks)["ce"]
dp=model.model.layers[L].mlp.down_proj
W=dp.weight; print("down_proj weight",W.shape)
mask=mx.ones((W.shape[1],)); 
import numpy as np
m=np.ones(W.shape[1],dtype=np.float32); m[NS]=0
dp.weight=W*mx.array(m)[None,:]
mx.eval(dp.weight)
a=T.eval_loss(model,blocks)["ce"]
r={"base":b,"ablated":a,"delta":a-b}
print(json.dumps(r)); json.dump(r,open("t1_mlx%s.json"%("_big" if os.environ.get("BIG") else ""),"w"))
