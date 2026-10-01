"""Load a Gemma 4 model with mlx-vlm and turn data records into model inputs."""
import json

import mlx.core as mx
from PIL import Image
from mlx_vlm import load as vlm_load


def load(path):
    model, processor = vlm_load(path)
    return model, processor


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def messages(rec, with_answer=True):
    content = []
    if rec.get("image"):
        content.append({"type": "image"})
    content.append({"type": "text", "text": rec["user"]})
    msgs = [{"role": "user", "content": content}]
    if with_answer and rec.get("assistant"):
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": rec["assistant"]}]})
    return msgs


def encode(processor, rec, with_answer=True, max_tokens=1024, answer_prefix=None):
    """Return (input_ids, kwargs) for one record. kwargs carries image tensors.
    answer_prefix: text appended after the generation prompt (start of the model's answer)."""
    has_answer = with_answer and bool(rec.get("assistant"))
    text = processor.apply_chat_template(
        messages(rec, with_answer), tokenize=False, add_generation_prompt=not has_answer
    )
    if answer_prefix and not has_answer:
        text = text + answer_prefix
    images = [Image.open(rec["image"]).convert("RGB")] if rec.get("image") else None
    out = processor(text=text, images=images, return_tensors="np")
    ids = mx.array(out["input_ids"])
    kwargs = {}
    for k in ("pixel_values", "image_position_ids", "mm_token_type_ids"):
        if k in out and out[k] is not None:
            kwargs[k] = mx.array(out[k])
    if ids.shape[1] > max_tokens and not images:
        ids = ids[:, :max_tokens]
        if "mm_token_type_ids" in kwargs:
            kwargs["mm_token_type_ids"] = kwargs["mm_token_type_ids"][:, :max_tokens]
    return ids, kwargs


def hidden(model, ids, kwargs):
    """Final normed hidden states. Passes image token types to the decoder, so image
    tokens get bidirectional attention. (mlx-vlm 0.7.3 gemma4_unified drops them.)"""
    emb = model.get_input_embeddings(input_ids=ids, **kwargs)
    return model.language_model.model(
        None, inputs_embeds=emb.inputs_embeds, per_layer_inputs=emb.per_layer_inputs,
        mm_token_type_ids=kwargs.get("mm_token_type_ids")
    )


def logprobs_at(model, h, positions, chunk=256):
    """Log-probabilities at the given positions, computed in chunks to bound memory."""
    out = []
    for i in range(0, len(positions), chunk):
        hc = h[0][mx.array(positions[i:i + chunk])]
        lg = model.language_model.logits_from_hidden(hc).astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        mx.eval(lp)
        out.append(lp)
    return mx.concatenate(out, axis=0)
