"""Check a nest model against LM Studio's 8-bit copy, then test width switching."""
import gc
import mlx.core as mx
from nest import elastic
from nest.io import load, read_jsonl, encode

recs = [r for r in read_jsonl("data/heldout.jsonl") if r["id"].split("-")[-1] in ("40", "41")][:20]

def run(model, proc):
    out = []
    for r in recs:
        ids, kw = encode(proc, r)
        lg = model(ids, **kw).logits[0].astype(mx.float32)
        out.append(lg - mx.logsumexp(lg, -1, keepdims=True)); mx.eval(out[-1])
    return out

def cmp(tag, ref, out):
    for r, a, b in zip(recs, ref, out):
        kl = (mx.exp(a) * (a - b)).sum(-1).mean().item()
        t1 = (a.argmax(-1) == b.argmax(-1)).astype(mx.float32).mean().item()
        print(f"{tag:10s} {r['id']:18s} KL={kl:.4f} top1={t1:.3f}", flush=True)

m, p = load("/Users/shonusengupta/.lmstudio/models/lmstudio-community/gemma-4-12B-it-MLX-8bit")
ref = run(m, p); del m; gc.collect(); mx.clear_cache()
elastic.patch()
m, p = load("models/nest8-id")
cmp("nest8-id", ref, run(m, p))
elastic.set_level(m, widths=[15360 // 2] * 48)
cmp("half-wid", ref, run(m, p))
print("peak GB", mx.get_peak_memory() / 1e9)
