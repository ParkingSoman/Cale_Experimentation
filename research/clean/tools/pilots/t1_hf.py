import os, glob, json, sys
ROOT="/Users/shonusengupta/Cale_Experimentation"
os.environ["HF_HOME"]=ROOT+"/research/tree/hf"; os.environ["HF_HUB_OFFLINE"]="1"
import torch, numpy as np
from nnsight import LanguageModel
from transformers import AutoTokenizer
d=glob.glob(ROOT+"/research/tree/hf/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/*")[0]
tok=AutoTokenizer.from_pretrained(d)
text=open(ROOT+"/research/tree/data/wikitext2_test.txt").read()
need=4*513; ids=[];pos=0
while len(ids)<need: ids+=tok.encode(text[pos:pos+200000]); pos+=200000
blocks=torch.tensor(ids[:need]).reshape(4,513)
x,y=blocks[:,:-1],blocks[:,1:]
lm=LanguageModel(d,dispatch=True,torch_dtype=torch.float32)
L,N=28,100
NS=list(range(0,300)) if os.environ.get('BIG') else [N]
def ce(logits): return torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]).float(),y.reshape(-1).to(logits.device)).item()
with torch.no_grad():
    with lm.trace(x):
        base=lm.output.logits.save()
    with lm.trace(x):
        dp=lm.model.layers[L].mlp.down_proj
        dp.input[:,:,NS]=0
        abl=lm.output.logits.save()
b,a=ce(base),ce(abl)
print(json.dumps({"base":b,"ablated":a,"delta":a-b}))
json.dump({"base":b,"ablated":a,"delta":a-b},open("t1_hf%s.json"%("_big" if os.environ.get("BIG") else "")+"","w"))
