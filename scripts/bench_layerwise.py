"""Step 1 gate: score the precision ladder's rungs before building the nest file.

Runs three versions of the 12B side by side, one decoder layer at a time
(the original bf16 does not fit in RAM). Scoring matches nest.gen.choose_letter:
the answer starts with "The answer is" and the best of " A".." J" wins.

| Stream | Layer weight tables                   | Norms, layer_scalar, final norm, vision ln/pos | Word table, vision tables |
|--------|---------------------------------------|------------------------------------------------|---------------------------|
| ocean  | original, original neuron order       | original                                       | original                  |
| r8     | row_scale * deq(A) + deq(B), dense    | original                                       | same rule as layers       |
| r4     | deq(A), dense (control vs qg4 bench)  | qg4 (= QAT)                                    | qg4's quantized modules   |

r8 and r4 use qg4's neuron order (a permutation does not change the output).
Samples run in chunks bounded by total tokens. Results are saved after each chunk;
a rerun skips items already scored.

usage: bench_layerwise.py NAME [--tasks mmlupro1k,mmstar300] [--limit N] [--tokens 60000]
Writes results/bench/NAME.json and results/NAME-errors.json (per-table weight errors).
"""
import argparse
import json
import os
import time
from collections import defaultdict

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm import load as vlm_load

from nest.gen import ANSWER_PREFIX
from nest.io import encode, read_jsonl
from nest.planes import A_BITS, A_GROUP, deq, make_planes

ORIG = "models/gemma-4-12B-it-bf16"
QG4 = "models/qg4"
PREFIX = "language_model.model.layers."
STREAMS = ("ocean", "r8", "r4")
NONLAYER_Q = ("language_model.model.embed_tokens", "vision_embedder.patch_dense",
              "embed_vision.embedding_projection")

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--tasks", default="mmlupro1k,mmstar300")
ap.add_argument("--limit", type=int)
ap.add_argument("--tokens", type=int, default=60000)
args = ap.parse_args()

mx.set_cache_limit(1024**3)
t0 = time.time()
model, proc = vlm_load(ORIG, lazy=True)
qmodel, _ = vlm_load(QG4, lazy=True)
lm = model.language_model.model
tk = proc.tokenizer
letter_ids = {c: tk.encode(" " + c, add_special_tokens=False)[0] for c in "ABCDEFGHIJ"}
assert all(len(tk.encode(" " + c, add_special_tokens=False)) == 1 for c in "ABCDEFGHIJ")
perms = np.load("results/perm-wanda.npz")


def layer_files(src):
    wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    out = defaultdict(set)
    for k, f in wm.items():
        if k.startswith(PREFIX):
            out[int(k[len(PREFIX):].split(".")[0])].add(f)
    return out


LF = {ORIG: layer_files(ORIG), QG4: layer_files(QG4)}


def fresh(src, i):
    """Lazy (not yet read) parameters of layer i, flat, keys relative to the layer."""
    p = f"{PREFIX}{i}."
    out = {}
    for f in LF[src][i]:
        for k, v in mx.load(f"{src}/{f}").items():
            if k.startswith(p):
                out[k[len(p):]] = v
    return out


errors = {}


def layer_params(i):
    o, g = fresh(ORIG, i), fresh(QG4, i)
    p = mx.array(perms[f"layer_{i}"])
    sets = {"ocean": o, "r8": {}, "r4": {}}
    for k, v in o.items():
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base and f"{base}.scales" in g:
            w = v[p] if ("gate_proj" in k or "up_proj" in k) else (v[:, p] if "down_proj" in k else v)
            pl = make_planes(w, (g[k], g[f"{base}.scales"], g[f"{base}.biases"]))
            errors[f"{i}.{base}"] = {e: pl[e] for e in ("err4", "err8", "err16")}
            sets["r8"][k] = pl["r8"].astype(mx.bfloat16)
            sets["r4"][k] = pl["a"].astype(mx.bfloat16)
            mx.eval(sets["r8"][k], sets["r4"][k])
        else:
            sets["r8"][k] = v
            sets["r4"][k] = g[k]
    return sets


# Non-layer parameters per stream (word table, vision tables, final norm, vision norms).
ORIG_NL = {k: v for k, v in tree_flatten(model.parameters()) if ".layers." not in k and "embed_audio" not in k}
QG4_NL = {k: v for k, v in tree_flatten(qmodel.parameters()) if ".layers." not in k and "embed_audio" not in k}
_r8_cache = {}


def _parent(root, path):
    *head, name = path.split(".")
    for h in head:
        root = getattr(root, h)
    return root, name


ORIG_MODS = {p: getattr(*_parent(model, p)) for p in NONLAYER_Q}
QG4_MODS = {p: getattr(*_parent(qmodel, p)) for p in NONLAYER_Q}


def nonlayer(stream):
    """Parameter values to load for a stream. r4 uses qg4's own quantized modules for
    the word table and vision tables (swapped in by set_nonlayer), so no dense copy."""
    if stream == "ocean":
        return dict(ORIG_NL)
    if stream == "r4":
        return {k: v for k, v in QG4_NL.items() if not k.startswith(NONLAYER_Q)}
    if _r8_cache:
        return dict(_r8_cache)
    out = {}
    for k, v in ORIG_NL.items():
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base in NONLAYER_Q:
            a_parts = (QG4_NL[k], QG4_NL[f"{base}.scales"], QG4_NL[f"{base}.biases"])
            rows, e4, e8, e16 = [], [], [], []
            step = 16384
            for r in range(0, v.shape[0], step):
                pl = make_planes(v[r:r + step], tuple(x[r:r + step] for x in a_parts))
                rows.append(pl["r8"].astype(mx.bfloat16))
                mx.eval(rows[-1])
                e4.append(pl["err4"]), e8.append(pl["err8"]), e16.append(pl["err16"])
                del pl
            out[k] = mx.concatenate(rows)
            mx.eval(out[k])
            del rows
            errors[base] = {"err4": float(np.mean(e4)), "err8": float(np.mean(e8)), "err16": float(np.mean(e16))}
        else:
            out[k] = v
    mx.eval(out)
    _r8_cache.update(out)
    return out


def set_nonlayer(stream):
    for p in NONLAYER_Q:
        parent, name = _parent(model, p)
        setattr(parent, name, QG4_MODS[p] if stream == "r4" else ORIG_MODS[p])
    model.update(tree_unflatten(list(nonlayer(stream).items())))


def prepare(recs):
    samples = []
    set_nonlayer("ocean")
    for r in recs:
        ids, kw = encode(proc, r, with_answer=False, max_tokens=4096, answer_prefix=ANSWER_PREFIX)
        samples.append({"rec": r, "ids": ids, "kw": kw, "h": {}})
    for stream in STREAMS:
        set_nonlayer(stream)
        for s in samples:
            emb = model.get_input_embeddings(input_ids=s["ids"], **s["kw"]).inputs_embeds
            mx.eval(emb)
            s["h"][stream] = emb
            if "masks" not in s:
                masks = lm._make_masks(emb, [None] * len(lm.layers), s["kw"].get("mm_token_type_ids"))
                s["masks"] = {layer.layer_type: m for layer, m in zip(lm.layers, masks)}
    return samples


def run(samples):
    for i, layer in enumerate(lm.layers):
        sets = layer_params(i)
        for stream in STREAMS:
            layer.update(tree_unflatten(list(sets[stream].items())))
            for s in samples:
                h, _, _ = layer(s["h"][stream], s["masks"][layer.layer_type], None)
                mx.eval(h)
                s["h"][stream] = h
        del sets
        layer.update(tree_unflatten(list(fresh(ORIG, i).items())))
        mx.clear_cache()
        if i % 8 == 7:
            print(f"  layer {i} {time.time() - t0:.0f}s peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


def score(samples, out, task):
    ids = mx.array(list(letter_ids.values()))
    for stream in STREAMS:
        set_nonlayer(stream)
        res = out.setdefault(stream, {}).setdefault(task, {"items": []})
        for s in samples:
            h = lm.norm(s["h"][stream][:, -1:])
            lg = model.language_model.logits_from_hidden(h)[0, 0].astype(mx.float32)
            lp = (lg - mx.logsumexp(lg))[ids]
            lp = dict(zip(letter_ids, np.array(lp).tolist()))
            r = s["rec"]
            pred = max(r["choices"], key=lambda c: lp[c])
            res["items"].append({"id": r["id"], "pred": pred, "ok": pred == r["answer"],
                                 "lp": {c: round(lp[c], 4) for c in r["choices"]}})
        res["n"] = len(res["items"])
        res["acc"] = sum(it["ok"] for it in res["items"]) / res["n"]


def chunks(recs):
    """Split into chunks whose total prompt tokens stay under --tokens."""
    cur, tok = [], 0
    for r in recs:
        ids, _ = encode(proc, r, with_answer=False, max_tokens=4096, answer_prefix=ANSWER_PREFIX)
        n = ids.shape[1]
        if cur and tok + n > args.tokens:
            yield cur
            cur, tok = [], 0
        cur.append(r)
        tok += n
    if cur:
        yield cur


os.makedirs("results/bench", exist_ok=True)
path = f"results/bench/{args.name}.json"
out = json.load(open(path)) if os.path.exists(path) else {}
for task in args.tasks.split(","):
    done = {it["id"] for it in out.get("ocean", {}).get(task, {}).get("items", [])}
    recs = [r for r in read_jsonl(f"data/bench/{task}.jsonl")[: args.limit] if r["id"] not in done]
    for ci, chunk in enumerate(chunks(recs)):
        samples = prepare(chunk)
        ntok = sum(s["ids"].shape[1] for s in samples)
        print(f"[{task}] chunk {ci}: {len(samples)} items, {ntok} tokens, {time.time() - t0:.0f}s", flush=True)
        run(samples)
        score(samples, out, task)
        json.dump(out, open(path, "w"), indent=1)
        json.dump(errors, open(f"results/{args.name}-errors.json", "w"), indent=1)
        print(f"[{task}] " + "  ".join(f"{s}={out[s][task]['acc']:.3f}" for s in STREAMS)
              + f" (n={out['ocean'][task]['n']}) {time.time() - t0:.0f}s peak {mx.get_peak_memory() / 1e9:.1f} GB",
              flush=True)
        del samples
        mx.clear_cache()
print("done", time.time() - t0)
