import t2_stubs, asyncio
from delphi.clients.client import Client, Response
from delphi.clients import OpenRouter
from delphi.latents import LatentCache, LatentDataset, LatentRecord
from delphi.latents.latents import ActivatingExample, NonActivatingExample, Latent
from delphi.explainers import DefaultExplainer
from delphi.scorers import DetectionScorer, FuzzingScorer
import torch

class FakeClient(Client):
    def __init__(self): super().__init__("fake"); self.calls=[]
    async def generate(self, prompt, **kw):
        self.calls.append((prompt,kw))
        if len(self.calls)==1: return Response(text="thinking\n[EXPLANATION]: the word 'the'")
        return Response(text="[1]")
toks=torch.arange(32); 
def ex(a): return ActivatingExample(tokens=toks, activations=a, normalized_activations=(a/a.max()*10).floor(), str_tokens=[f"t{i}" for i in range(32)])
a=torch.zeros(32); a[5]=3.0
rec=LatentRecord(Latent("model.layers.28.mlp.down_proj",100))
rec.train=[ex(a) for _ in range(3)]
rec.test=[ex(a)]; rec.not_active=[NonActivatingExample(tokens=toks,activations=torch.zeros(32),str_tokens=[f"t{i}" for i in range(32)])]
c=FakeClient()
async def main():
    r=await DefaultExplainer(c, threshold=0.3)(rec)
    print("explanation:",r.explanation)
    rec.explanation=r.explanation
    s=await DetectionScorer(c, n_examples_shown=1)(rec)
    print("detection results:",[ (o.prediction,o.correct) for o in s.score])
    print("prompt roles:",[m["role"] for m in c.calls[0][0]])
asyncio.run(main())
print("OpenRouter ctor ok:", OpenRouter("anthropic/claude-3.5-haiku", api_key="x").url)
